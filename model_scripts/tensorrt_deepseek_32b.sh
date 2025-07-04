#!/bin/bash 
trtllm-eval --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --backend tensorrt --max_seq_len 4096  gsm8k
