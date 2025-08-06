#!/bin/bash 
trtllm-eval --model "/data/lm-evaluation-harness/Llama-3.1-8B-Instruct" --backend tensorrt --max_seq_len 4096  cnn_dailymail