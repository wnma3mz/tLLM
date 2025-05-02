#!/bin/bash
# master 的地址，请求分配模型的节点
MASTER_URL=http://mac-mini:8022

# export MPI_BUFFER_SIZE=1048576
export PYTHONPATH="/Users/lujianghu/Documents/tLLM":$PYTHONPATH

mpirun -np 1 \
    --hostfile mpi_hosts \
    -wdir /Users/lujianghu/Documents/tLLM \
    -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ \
    -x PYTHONPATH \
    --mca btl_tcp_links 128 \
    /opt/homebrew/bin/python3 tllm/grpc/worker_service/worker_server.py --master_addr $MASTER_URL --is_debug