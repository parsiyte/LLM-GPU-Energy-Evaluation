#!/bin/bash
lm_eval --model vllm \
    --model_args pretrained=EleutherAI/pythia-70m,dtype=float  \
    --tasks triviaqa \
    --device cuda:0 \
    --batch_size  auto

exit 0




