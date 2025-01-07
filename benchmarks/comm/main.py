from concurrent import futures
from datetime import datetime
import statistics
import time
import zlib

import grpc
import lz4.frame
import tabulate
import torch

from tllm.commons.convert import Convertor
from tllm.grpc.proto import schemas_pb2, schemas_pb2_grpc


def compress_bytes(tensor_proto: schemas_pb2.BFloat16Tensor) -> schemas_pb2.BFloat16Tensor:
    if not use_zlib and not use_lz4:
        return tensor_proto
    if use_zlib:
        tensor_proto.data = zlib.compress(tensor_proto.data)
    if use_lz4:
        tensor_proto.data = lz4.frame.compress(tensor_proto.data)
    return tensor_proto


def uncompress_bytes(tensor_proto: schemas_pb2.BFloat16Tensor) -> schemas_pb2.BFloat16Tensor:
    if not use_zlib and not use_lz4:
        return tensor_proto
    if use_zlib:
        tensor_proto.data = zlib.decompress(tensor_proto.data)
    if use_lz4:
        tensor_proto.data = lz4.frame.decompress(tensor_proto.data)
    return tensor_proto


class MatrixServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self):
        self.received_sizes = []
        self.processing_times = []

    def Forward(self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext):
        start_time = time.time()
        output = compress_bytes(uncompress_bytes(request.hidden_states))

        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed",
            status=200,
            output=output,
            cost_time=time.time() - start_time,
        )


class PerformanceTester:
    def __init__(self, host="localhost", port=50051):
        self.host = host
        self.port = port
        self.server = None
        self.servicer = None

    def start_server(self):
        self.servicer = MatrixServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()

    def stop_server(self):
        if self.server:
            self.server.stop(0)

    def run_client_test(self, matrix_shape, num_iterations=3):
        tensor = torch.randn(*matrix_shape)
        size_mb = tensor.nbytes / (1024 * 1024)

        convertor = Convertor()
        byte_tensor = convertor.serialize(tensor)
        transmission_times = []
        calc_times = []

        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = schemas_pb2_grpc.RPCServiceStub(channel)
            request = schemas_pb2.ForwardRequest(uuid=["123"], seq_len=[1], hidden_states=compress_bytes(byte_tensor))

            for _ in range(num_iterations):
                start_time = time.time()
                response = stub.Forward(request)
                transmission_time = time.time() - start_time - response.cost_time
                transmission_times.append(transmission_time)
                calc_times.append(response.cost_time)

        return {
            "matrix_shape": matrix_shape,
            "size_mb": size_mb,
            "transmission_times": transmission_times,
            "calc_times": calc_times,
        }

    def run_performance_test(self, test_shapes=None):
        if test_shapes is None:
            test_shapes = [
                # (1, 8192),
                # (4, 8192),
                # (16, 8192),
                (32, 8192),
                # (64, 8192),
            ]

        print(f"\n=== Starting gRPC Matrix Performance Test (use_zlib: {use_zlib}; use_lz4: {use_lz4}) ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.start_server()
        results = []
        table_data = []
        try:
            for shape in test_shapes:
                result = self.run_client_test(shape, 100)
                results.append(result)

                times = result["transmission_times"]
                size = result["size_mb"]
                mean_time = statistics.mean(times)
                calc_time = statistics.mean(result["calc_times"])
                # np.percentile(times, 95) * 1000 # P95 时间
                table_data.append(
                    [shape, mean_time * 1000, calc_time * 1000, (mean_time + calc_time) * 1000, size / mean_time]
                )
        finally:
            self.stop_server()

        print(
            tabulate.tabulate(
                table_data,
                headers=["Matrix Shape", "Transmission(ms)", "Compress(ms)", "Total(ms)", "Throughput(MB/s)"],
                tablefmt="grid",
                floatfmt=".2f",
            )
        )

        return results


def main():
    tester = PerformanceTester("192.168.1.4")
    tester.run_performance_test()


if __name__ == "__main__":
    use_zlib = False
    use_lz4 = False
    if use_zlib and use_lz4:
        raise ValueError("Cannot use both zlib and lz4 at the same time.")
    main()
