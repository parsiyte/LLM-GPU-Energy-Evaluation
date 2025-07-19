#!/bin/bash
lm_eval   --model vllm   --model_args pretrained="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,add_bos_token=True,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1   --tasks gsm8k  --batch_size auto
exit 0
