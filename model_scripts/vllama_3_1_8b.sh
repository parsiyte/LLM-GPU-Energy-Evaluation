#!/bin/bash
lm_eval   --model vllm   --model_args pretrained="/data/lm-evaluation-harness/Llama-3.1-8B-Instruct",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 --fewshot_as_multiturn --num_fewshot 3   --tasks bbh   --batch_size auto
exit 0
