# LLM GPU Energy Evaluation

These scripts are used for conducting the experiments and collecting the results for the paper on LLM GPU energy evaluation.

## Requirements

To run these experiments, you will need to have the DEPO tool. Please make sure to clone or copy the `split` directory from the DEPO repository into the root of this repository folder. The scripts expect the following directory structure:


\`\`\`
.
├── experiment_runners/
├── model_scripts/
├── split/
│   ├── build/
│   └── profiling_injection/
├── prepare_depo.sh
├── ...
└── README.md
\`\`\` 
