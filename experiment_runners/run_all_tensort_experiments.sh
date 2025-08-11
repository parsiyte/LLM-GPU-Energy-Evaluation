#!/bin/bash

# This script runs all experiments in a predefined order.

# Exit immediately if a command exits with a non-zero status.
set -e


# --- Run TensorRT Deepseek Experiments ---
echo "Running TensorRT Deepseek Experiments... SKIPPED"
./run_tensorrt_deepseek_experiments.sh

# --- Run TensorRT Llama Experiment (A100) ---
echo "Running TensorRT Llama Experiment (A100)... SKIPPED"
./run_tensort_a100_llama_experiment.sh

# --- Run TensorRT Llama Experiment (A4500) ---
echo "Running TensorRT Llama Experiment (A4500)... SKIPPED"
./run_tensorrt_a4500_llama_experiment.sh

echo "All experiments completed successfully."
