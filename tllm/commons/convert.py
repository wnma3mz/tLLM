from typing import List

import lz4.frame
import numpy as np

from tllm import BACKEND, CONVERT_DTYPE, DES_DTYPE, DTYPE, BackendEnum
from tllm.grpc.proto import schemas_pb2, schemas_pb2_grpc
from tllm.schemas import MIX_TENSOR

if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    serialize_func = lambda tensor, dtype: bytes(tensor.astype(dtype))
    deserialize_func = lambda proto, x, dtype, des_dtype: mx.array(
        np.frombuffer(x, dtype=des_dtype), dtype=dtype
    ).reshape(*proto.shape)
else:
    import torch

    serialize_func = lambda tensor, tensor_dtype: tensor.to(tensor_dtype).detach().cpu().numpy().tobytes()
    deserialize_func = (
        lambda proto, x, dtype, des_dtype: torch.frombuffer(np.copy(x), dtype=des_dtype).to(dtype).view(*proto.shape)
    )


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


class Convertor:
    def __init__(self, ser_dtype=CONVERT_DTYPE, des_dtype=DES_DTYPE, dtype=CONVERT_DTYPE):
        self.serialize_func = serialize_func
        self.deserialize_func = deserialize_func
        self.ser_dtype = ser_dtype
        self.des_dtype = des_dtype
        self.dtype = dtype

    def serialize(self, tensor: MIX_TENSOR) -> schemas_pb2.BFloat16Tensor:
        # TODO: support bfloat16
        tensor_proto = schemas_pb2.BFloat16Tensor()
        # seq_len x hidden_size
        tensor_proto.shape.extend(tensor.shape)
        tensor_bytes = self.serialize_func(tensor, self.ser_dtype)
        flag = tensor.shape[0] >= 64
        tensor_proto.data = lz4.frame.compress(tensor_bytes) if flag else tensor_bytes
        return tensor_proto

    def deserialize(self, tensor_proto: schemas_pb2.BFloat16Tensor) -> MIX_TENSOR:
        flag = tensor_proto.shape[0] >= 64
        tensor_bytes = lz4.frame.decompress(tensor_proto.data) if flag else tensor_proto.data
        return self.deserialize_func(tensor_proto, tensor_bytes, self.dtype, self.des_dtype)
