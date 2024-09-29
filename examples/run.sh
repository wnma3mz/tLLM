PORT1=25001
PORT2=$(($PORT1+1))
MASTER_PORT1=29502
MASTER_PORT2=$(($MASTER_PORT1+1))

MODEL_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0
WEIGHT_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0/master_weight.pt
CONFIG_PATH=./examples/config.json

export OMP_NUM_THREADS=8;


PYTHONPATH="./tllm":$PYTHONPATH python3 grpc_exp/app.py --port 8000 --model_path $MODEL_PATH --weight_path $WEIGHT_PATH --config_path $CONFIG_PATH

# 使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行，通信时仅通信输入

# curl http://localhost:8000/v1/chat/completions -X POST \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer common" \
#   -d '{
#     "messages": [
#       {
#         "role": "user",
#         "content": "Hello, how are you?"
#       }
#     ]
#   }'
