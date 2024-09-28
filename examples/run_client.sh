MASTER_PORT=29501
GRPC_PORT=25001
export OMP_NUM_THREADS=8;
PYTHONPATH="./src2:./src":$PYTHONPATH; export OMP_NUM_THREADS=8; torchrun --nproc_per_node=2 --master_port=$MASTER_PORT src2/rpc_comm/client.py --port=$GRPC_PORT --start_layer_idx=0 --end_layer_idx=11 > client_$GRPC_PORT.log 2>&1 &
PYTHONPATH="./src2:./src":$PYTHONPATH; export OMP_NUM_THREADS=8; torchrun --nproc_per_node=2 --master_port=$(($MASTER_PORT+2)) src2/rpc_comm/client.py --port=$(($GRPC_PORT+2)) --start_layer_idx=11 --end_layer_idx=22 > client_$(($GRPC_PORT+2)).log 2>&1 &