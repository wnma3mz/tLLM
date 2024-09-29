PORT=29605
MODEL_PATH="/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
# 由于 torch 的原因输出并不会到日志里
# 确保每个 node 与 master 的通信良好
PYTHONPATH="./torch_rpc_exp":$PYTHONPATH python3 torch_rpc_exp/worker.py --world_size 5 --pp_start 1 --pp_end 2 --rpc_port $PORT --process_port $(($PORT+2)) > run_pp0.log 2>&1 &
PYTHONPATH="./torch_rpc_exp":$PYTHONPATH python3 torch_rpc_exp/worker.py --world_size 5 --pp_start 3 --pp_end 4 --rpc_port $PORT --process_port $(($PORT+3)) > run_pp1.log 2>&1 &
PYTHONPATH="./torch_rpc_exp":$PYTHONPATH python3 torch_rpc_exp/client.py --world_size 5 --rpc_port $PORT --model_path $MODEL_PATH