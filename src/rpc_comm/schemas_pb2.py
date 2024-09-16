# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/rpc_comm/schemas.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1asrc/rpc_comm/schemas.proto\x12\x07schemas\x1a\x19google/protobuf/any.proto\x1a\x1cgoogle/protobuf/struct.proto"\x19\n\x05\x41rray\x12\x10\n\x08\x65lements\x18\x01 \x03(\x02"&\n\x06Matrix\x12\x1c\n\x04rows\x18\x01 \x03(\x0b\x32\x0e.schemas.Array")\n\x06Tensor\x12\x1f\n\x06layers\x18\x01 \x03(\x0b\x32\x0f.schemas.Matrix".\n\x0b\x42lockTensor\x12\x1f\n\x06\x62locks\x18\x01 \x03(\x0b\x32\x0f.schemas.Tensor"\xbb\x01\n\x15MultiDimensionalArray\x12\x1f\n\x05\x61rray\x18\x01 \x01(\x0b\x32\x0e.schemas.ArrayH\x00\x12!\n\x06matrix\x18\x02 \x01(\x0b\x32\x0f.schemas.MatrixH\x00\x12!\n\x06tensor\x18\x03 \x01(\x0b\x32\x0f.schemas.TensorH\x00\x12,\n\x0c\x62lock_tensor\x18\x04 \x01(\x0b\x32\x14.schemas.BlockTensorH\x00\x42\r\n\x0bmulti_array"\xaa\x01\n\x0bLayerConfig\x12\'\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\x17.google.protobuf.Struct\x12\x17\n\x0flayer_idx_start\x18\x02 \x01(\x05\x12\x15\n\rlayer_idx_end\x18\x03 \x01(\x05\x12\x13\n\x0btp_url_list\x18\x04 \x03(\t\x12\x0f\n\x07tp_size\x18\x05 \x01(\x05\x12\x1c\n\x14layer_state_dict_dir\x18\x06 \x01(\t"R\n\x0b\x46orwardData\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x35\n\rhidden_states\x18\x02 \x01(\x0b\x32\x1e.schemas.MultiDimensionalArray"\x88\x02\n\tMLPConfig\x12\x12\n\ninput_size\x18\x01 \x01(\x05\x12\x13\n\x0boutput_size\x18\x02 \x01(\x05\x12\x10\n\x08mlp_bias\x18\x03 \x01(\x08\x12\x11\n\tproj_name\x18\x04 \x01(\t\x12\x11\n\tlayer_idx\x18\x05 \x01(\x05\x12\x0e\n\x06tp_idx\x18\x06 \x01(\x05\x12\x0f\n\x07tp_size\x18\x07 \x01(\x05\x12\x17\n\x0fstate_dict_path\x18\x08 \x01(\t\x12)\n\x0bweight_data\x18\t \x01(\x0b\x32\x14.google.protobuf.Any\x12\'\n\tbias_data\x18\n \x01(\x0b\x32\x14.google.protobuf.Any\x12\x0c\n\x04name\x18\x0b \x01(\t"\x9e\x01\n\x0eMLPForwardData\x12\x11\n\tproj_name\x18\x01 \x01(\t\x12\x0e\n\x06tp_idx\x18\x02 \x01(\x05\x12\x11\n\tlayer_idx\x18\x03 \x01(\x05\x12\x35\n\rhidden_states\x18\x04 \x01(\x0b\x32\x1e.schemas.MultiDimensionalArray\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x11\n\tcost_time\x18\x06 \x01(\x02"-\n\x0eStatusResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\x05"q\n\x0f\x46orwardResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\x05\x12.\n\x06output\x18\x03 \x01(\x0b\x32\x1e.schemas.MultiDimensionalArray\x12\x11\n\tcost_time\x18\x04 \x01(\x02"-\n\x0eHealthResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\x05".\n\x0fMLPKeysResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\x05"4\n\x15InitModelFlagResponse\x12\x0b\n\x03msg\x18\x01 \x01(\x08\x12\x0e\n\x06status\x18\x02 \x01(\x05"\x07\n\x05\x45mpty2\xa5\x03\n\nRPCService\x12:\n\tInitModel\x12\x14.schemas.LayerConfig\x1a\x17.schemas.StatusResponse\x12\x39\n\x07\x46orward\x12\x14.schemas.ForwardData\x1a\x18.schemas.ForwardResponse\x12\x36\n\x07InitMLP\x12\x12.schemas.MLPConfig\x1a\x17.schemas.StatusResponse\x12?\n\nForwardMLP\x12\x17.schemas.MLPForwardData\x1a\x18.schemas.ForwardResponse\x12\x31\n\x06Health\x12\x0e.schemas.Empty\x1a\x17.schemas.HealthResponse\x12\x33\n\x07MLPKeys\x12\x0e.schemas.Empty\x1a\x18.schemas.MLPKeysResponse\x12?\n\rInitModelFlag\x12\x0e.schemas.Empty\x1a\x1e.schemas.InitModelFlagResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "src.rpc_comm.schemas_pb2", _globals
)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_ARRAY"]._serialized_start = 96
    _globals["_ARRAY"]._serialized_end = 121
    _globals["_MATRIX"]._serialized_start = 123
    _globals["_MATRIX"]._serialized_end = 161
    _globals["_TENSOR"]._serialized_start = 163
    _globals["_TENSOR"]._serialized_end = 204
    _globals["_BLOCKTENSOR"]._serialized_start = 206
    _globals["_BLOCKTENSOR"]._serialized_end = 252
    _globals["_MULTIDIMENSIONALARRAY"]._serialized_start = 255
    _globals["_MULTIDIMENSIONALARRAY"]._serialized_end = 442
    _globals["_LAYERCONFIG"]._serialized_start = 445
    _globals["_LAYERCONFIG"]._serialized_end = 615
    _globals["_FORWARDDATA"]._serialized_start = 617
    _globals["_FORWARDDATA"]._serialized_end = 699
    _globals["_MLPCONFIG"]._serialized_start = 702
    _globals["_MLPCONFIG"]._serialized_end = 966
    _globals["_MLPFORWARDDATA"]._serialized_start = 969
    _globals["_MLPFORWARDDATA"]._serialized_end = 1127
    _globals["_STATUSRESPONSE"]._serialized_start = 1129
    _globals["_STATUSRESPONSE"]._serialized_end = 1174
    _globals["_FORWARDRESPONSE"]._serialized_start = 1176
    _globals["_FORWARDRESPONSE"]._serialized_end = 1289
    _globals["_HEALTHRESPONSE"]._serialized_start = 1291
    _globals["_HEALTHRESPONSE"]._serialized_end = 1336
    _globals["_MLPKEYSRESPONSE"]._serialized_start = 1338
    _globals["_MLPKEYSRESPONSE"]._serialized_end = 1384
    _globals["_INITMODELFLAGRESPONSE"]._serialized_start = 1386
    _globals["_INITMODELFLAGRESPONSE"]._serialized_end = 1438
    _globals["_EMPTY"]._serialized_start = 1440
    _globals["_EMPTY"]._serialized_end = 1447
    _globals["_RPCSERVICE"]._serialized_start = 1450
    _globals["_RPCSERVICE"]._serialized_end = 1871
# @@protoc_insertion_point(module_scope)
