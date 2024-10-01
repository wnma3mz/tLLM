MASTER_PORT=29501
GRPC_PORT=25001
MODEL_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0
TP=2

export OMP_NUM_THREADS=8;
export PYTHONPATH="./tllm":$PYTHONPATH;

# torchrun --nproc_per_node=$TP --master_port=$MASTER_PORT tllm/rpc/client.py --port=$GRPC_PORT --start_layer_idx=0 --end_layer_idx=11 --model_path $MODEL_PATH > grpc_$GRPC_PORT.log 2>&1 &
# torchrun --nproc_per_node=$TP --master_port=$(($MASTER_PORT+2)) tllm/rpc/client.py --port=$(($GRPC_PORT+2)) --start_layer_idx=11 --end_layer_idx=22 --model_path $MODEL_PATH > grpc_$(($GRPC_PORT+2)).log 2>&1 &

python3 tllm/rpc/client.py --port=$GRPC_PORT --start_layer_idx=0 --end_layer_idx=11 --model_path $MODEL_PATH
