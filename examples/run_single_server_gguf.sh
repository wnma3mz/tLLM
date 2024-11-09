#!/bin/bash
BASE_PATH=/Users/lujianghu/Documents/

MODEL_PATH=/Users/jianghulu/Downloads/meta-llama-3.1-8b-instruct-abliterated.Q8_0.gguf

export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;


python3 -m tllm.entrypoints.api_server --port 8022 --model_path $MODEL_PATH --is_local