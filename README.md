# LLM GPU Energy Evaluation

These scripts are used for conducting the experiments and collecting the results for the paper on LLM GPU energy evaluation.

## Requirements

To run these experiments, you will need to have the DEPO tool. Please make sure to clone or copy the `split` directory from the DEPO repository into the root of this repository folder. 

Here is the folder structer:

.
├── README.md
├── experiment_runners
│   ├── run_deepseek_32b_experiment.sh
│   ├── run_llama_experiment.sh
│   ├── run_mistrial_experiment_a100.sh
│   ├── run_pythia_experiments.sh
│   ├── run_quantized_llama_experiments.sh
│   ├── run_qwen3_30b_3a_experiment.sh
│   ├── run_tensorrt_deepseek_experiments.sh
│   └── run_tensorrt_llama_experiment.sh
├── model_scripts
│   ├── int4_vllama_3_1_8b.sh
│   ├── int8_vllama_3_1_8b.sh
│   ├── int8_w16_vllama_3_1_8b.sh
│   ├── pythia_12b.sh
│   ├── pythia_14m.sh
│   ├── pythia_160m.sh
│   ├── pythia_1_4b.sh
│   ├── pythia_1b.sh
│   ├── pythia_2_8b.sh
│   ├── pythia_410m.sh
│   ├── pythia_6_9b.sh
│   ├── pythia_70m.sh
│   ├── qwen3_30b_3a.sh
│   ├── tensorrt_a4500_llama_3_1_8b.sh
│   ├── tensorrt_deepseek_32b.sh
│   ├── tensorrt_llama_3_1_8b.sh
│   ├── v_deepseek_32b.sh
│   ├── v_mistral_7b.sh
│   └── vllama_3_1_8b.sh
├── prepare_depo.sh
└── prepare_tensorrt.sh
