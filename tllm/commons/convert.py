from typing import List

import lz4.frame
import numpy as np

from tllm import BACKEND, BackendEnum
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.schemas_pb2 import BFloat16Tensor
from tllm.schemas import MIX_TENSOR

if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    serialize_func = lambda tensor: bytes(tensor.astype(mx.float16))
    deserialize_func = lambda x: mx.array(np.frombuffer(x[1], dtype=np.float16), dtype=mx.bfloat16).reshape(*x[0].shape)
else:
    import torch

    serialize_func = lambda tensor: tensor.to(torch.float16).cpu().detach().numpy().tobytes()
    deserialize_func = lambda x: torch.frombuffer(x[1], dtype=torch.float16).to(torch.bfloat16).view(*x[0].shape)


def protobuf_to_list(proto_message):
    if proto_message.HasField("block_tensor") and proto_message.block_tensor is not None:
        return [
            [[list(row.elements) for row in matrix.rows] for matrix in tensor.layers]
            for tensor in proto_message.block_tensor.blocks
        ]
    elif proto_message.HasField("tensor") and proto_message.tensor is not None:
        return [[list(row.elements) for row in matrix.rows] for matrix in proto_message.tensor.layers]
    elif proto_message.HasField("matrix") and proto_message.matrix is not None:
        return [list(row.elements) for row in proto_message.matrix.rows]
    elif proto_message.HasField("array") and proto_message.array is not None:
        return list(proto_message.array.elements)
    else:
        raise ValueError("Unknown multi-dimensional array type in protobuf message.")


def list_to_protobuf(data: List):
    multi_array_proto = schemas_pb2.MultiDimensionalArray()

    if isinstance(data, list):
        # Check the dimensionality of the list
        if all(isinstance(item, float) for item in data):
            # One-dimensional array
            array = schemas_pb2.Array()
            array.elements.extend(data)
            multi_array_proto.array.CopyFrom(array)
        elif all(isinstance(row, list) and all(isinstance(item, float) for item in row) for row in data):
            # Two-dimensional array (Matrix)
            matrix = schemas_pb2.Matrix()
            for row in data:
                array = schemas_pb2.Array()
                array.elements.extend(row)
                matrix.rows.add().CopyFrom(array)
            multi_array_proto.matrix.CopyFrom(matrix)
        elif all(
            isinstance(layer, list)
            and all(isinstance(row, list) and all(isinstance(item, float) for item in row) for row in layer)
            for layer in data
        ):
            # Three-dimensional array (Tensor)
            tensor = schemas_pb2.Tensor()
            for layer in data:
                matrix = schemas_pb2.Matrix()
                for row in layer:
                    array = schemas_pb2.Array()
                    array.elements.extend(row)
                    matrix.rows.add().CopyFrom(array)
                tensor.layers.add().CopyFrom(matrix)
            multi_array_proto.tensor.CopyFrom(tensor)
        elif all(
            isinstance(block, list)
            and all(
                isinstance(layer, list)
                and all(isinstance(row, list) and all(isinstance(item, float) for item in row) for row in layer)
                for layer in block
            )
            for block in data
        ):
            # Four-dimensional array (BlockTensor)
            block_tensor = schemas_pb2.BlockTensor()
            for block in data:
                tensor = schemas_pb2.Tensor()
                for layer in block:
                    matrix = schemas_pb2.Matrix()
                    for row in layer:
                        array = schemas_pb2.Array()
                        array.elements.extend(row)
                        matrix.rows.add().CopyFrom(array)
                    tensor.layers.add().CopyFrom(matrix)
                block_tensor.blocks.add().CopyFrom(tensor)
            multi_array_proto.block_tensor.CopyFrom(block_tensor)
        else:
            raise ValueError("Unsupported list structure. Make sure the list dimensions match the protobuf structure.")
    else:
        raise ValueError("Input data must be a list.")

    return multi_array_proto


def serialize_tensor(tensor: MIX_TENSOR) -> BFloat16Tensor:
    # TODO: support bfloat16
    tensor_proto = BFloat16Tensor()
    # seq_len x hidden_size
    tensor_proto.shape.extend(tensor.shape)
    tensor_bytes = serialize_func(tensor)
    flag = tensor.shape[0] >= 64
    tensor_proto.data = lz4.frame.compress(tensor_bytes) if flag else tensor_bytes
    return tensor_proto


def deserialize_tensor(tensor_proto: BFloat16Tensor) -> MIX_TENSOR:
    flag = tensor_proto.shape[0] >= 64
    tensor_bytes = lz4.frame.decompress(tensor_proto.data) if flag else tensor_proto.data
    return deserialize_func((tensor_proto, tensor_bytes))
