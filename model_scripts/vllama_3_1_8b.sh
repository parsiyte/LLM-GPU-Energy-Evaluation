#!/bin/bash
lm_eval   --model vllm   --model_args pretrained="/data/lm-evaluation-harness/Llama-3.1-8B-Instruct",dtype=auto,max_model_len=4096,tensor_parallel_size=1   --tasks gsm8k --apply_chat_template --fewshot_as_multiturn  --num_fewshot 8 --gen_kwargs temperature=0  --batch_size auto
exit 0
