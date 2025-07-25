#!/bin/bash 
trtllm-eval --model "/data/lm-evaluation-harness/DeepSeek-R1-Distill-Qwen-32B" --backend tensorrt --max_seq_len 4096  gsm8k
