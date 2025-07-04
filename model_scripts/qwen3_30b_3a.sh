#!/bin/bash
lm_eval \
  --model vllm \
  --model_args pretrained="/data/lm-evaluation-harness/Qwen3-30B-A3B",dtype=auto,add_bos_token=False,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks  gsm8k \
  --batch_size auto
