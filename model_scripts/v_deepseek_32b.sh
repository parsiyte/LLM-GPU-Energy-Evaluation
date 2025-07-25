#!/bin/bash
lm_eval \
  --model vllm \
  --model_args pretrained="/data/lm-evaluation-harness/DeepSeek-R1-Distill-Qwen-32B",dtype=auto,max_model_len=4096,tensor_parallel_size=1,data_parallel_size=1,enable_chunked_prefill=True \
  --tasks gsm8k \
  --batch_size auto \

exit 0
