import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_addr", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--grpc_port", type=int, default=25001)
    parser.add_argument("--http_port", type=int, default=8022)
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--is_fake", action="store_true")
    return parser.parse_args()
