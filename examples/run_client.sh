MASTER_PORT=29501
GRPC_PORT=25001
MODE_SIZE=$1
TP=$2
PP=$3

# PP 当前仅限于 0 和 1
if [ $PP != "0" ] && [ $PP != "1" ]; then
    echo "Invalid pp size"
    exit 1
fi

if [ $MODE_SIZE == "1.1" ]; then
    MODEL_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0
    if [ $PP == "0" ]; then
        START_LAYER_IDX=0
        END_LAYER_IDX=11
    elif [ $PP == "1" ]; then
        START_LAYER_IDX=11
        END_LAYER_IDX=22
    else
        echo "Invalid pp size"
        exit 1
    fi
elif [ $MODE_SIZE == "1" ]; then
    MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
    if [ $PP == "0" ]; then
        START_LAYER_IDX=0
        END_LAYER_IDX=8
    elif [ $PP == "1" ]; then
        START_LAYER_IDX=8
        END_LAYER_IDX=16
    else
        echo "Invalid pp size"
        exit 1
    fi
elif [ $MODE_SIZE == "3" ]; then
    MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-3B-Instruct
    if [ $PP == "0" ]; then
        START_LAYER_IDX=0
        END_LAYER_IDX=14
    elif [ $PP == "1" ]; then
        START_LAYER_IDX=14
        END_LAYER_IDX=28
    else
        echo "Invalid pp size"
        exit 1
    fi
elif [ $MODE_SIZE == "8" ]; then
    MODEL_PATH=/Users/lujianghu/Documents/Meta-Llama-3-8B-Instruct
    if [ $PP == "0" ]; then
        START_LAYER_IDX=0
        END_LAYER_IDX=16
    elif [ $PP == "1" ]; then
        START_LAYER_IDX=16
        END_LAYER_IDX=32
    else
        echo "Invalid pp size"
        exit 1
    fi
elif [ $MODE_SIZE == "70" ]; then
    MODEL_PATH=/Users/lujianghu/Documents/Meta-Llama-3-70B-Instruct
else 
    echo "Invalid mode size"
    exit 1
fi

export OMP_NUM_THREADS=8;
export PYTHONPATH="./tllm":$PYTHONPATH;

if [ $PP == "0" ]; then
    torchrun --nproc_per_node=$TP --master_port=$MASTER_PORT tllm/rpc/client.py --port=$GRPC_PORT --start_layer_idx=$START_LAYER_IDX --end_layer_idx=$END_LAYER_IDX --model_path $MODEL_PATH
elif [ $PP == "1" ]; then
    torchrun --nproc_per_node=$TP --master_port=$(($MASTER_PORT+1)) tllm/rpc/client.py --port=$(($GRPC_PORT+1)) --start_layer_idx=$START_LAYER_IDX --end_layer_idx=$END_LAYER_IDX --model_path $MODEL_PATH
fi