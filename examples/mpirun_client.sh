#!/bin/bash
# master 的地址，请求分配模型的节点
MASTER_URL=http://mac-mini:8022

# mpirun -n 2 /opt/homebrew/bin/python3 tllm/grpc/worker_service/worker_server.py --master_addr $MASTER_URL --is_debug
# mpirun --verbose -wdir /Users/lujianghu/Documents/tLLM -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ -x PYTHONPATH="/Users/lujianghu/Documents/tLLM":$PYTHONPATH /opt/homebrew/bin/python3 tllm/grpc/worker_service/worker_server.py --master_addr $MASTER_URL --is_debug

# export MPI_BUFFER_SIZE=1048576
mpirun --verbose -hostfile mpi_hosts -wdir /Users/lujianghu/Documents/tLLM -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ -x PYTHONPATH="/Users/lujianghu/Documents/tLLM":$PYTHONPATH /opt/homebrew/bin/python3 tllm/grpc/worker_service/worker_server.py --master_addr $MASTER_URL
#  --is_debug

# mpirun --verbose -hostfile mpi_hosts -wdir /Users/lujianghu/Documents/tLLM -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ -x PYTHONPATH="/Users/lujianghu/Documents/tLLM":$PYTHONPATH /opt/homebrew/bin/python3 test_mpi4.py

# tllm.client --master_addr $MASTER_URL --is_debug --config examples/config_one.json --client_idx 0