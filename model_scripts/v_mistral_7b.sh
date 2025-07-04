#!/bin/bash
lm_eval \
  --model vllm \
  --model_args pretrained="mistralai/Mistral-7B-Instruct-v0.3",dtype=auto,max_model_len=4096,tensor_parallel_size=1,add_bos_token=True \
  --tasks gsm8k \
  --batch_size auto \

exit 0
