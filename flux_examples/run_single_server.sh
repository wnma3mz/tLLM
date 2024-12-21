#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/flux/schnell_4bit

tllm.server --model_path $MODEL_PATH --client_size 1 --is_local --is_debug --is_image


