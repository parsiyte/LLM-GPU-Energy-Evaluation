#!/bin/bash

# This script runs all experiments in a predefined order.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Run Pythia Experiments ---
echo "Running Pythia Experiments..."
./run_pythia_experiments.sh

# --- Run Llama Experiment ---
echo "Running Llama Experiment..."
./run_llama_experiment.sh



# --- Run Quantized Llama Experiments ---
echo "Running Quantized Llama Experiments..."
./run_quantized_llama_experiments.sh

# --- Run Mistral Experiment on A100 ---
echo "Running Mistral Experiment on A100..."
./run_mistrial_experiment_a100.sh

# --- Run Deepseek 32B Experiment ---
echo "Running Deepseek 32B Experiment..."
./run_deepseek_32b_experiment.sh

# --- Run Qwen3 30B 3A Experiment ---
echo "Running Qwen3 30B 3A Experiment..."
./run_qwen3_30b_3a_experiment.sh

# --- Run TensorRT Deepseek Experiments ---
echo "Running TensorRT Deepseek Experiments..."
./run_tensorrt_deepseek_experiments.sh

# --- Run TensorRT Llama Experiment ---
echo "Running TensorRT Llama Experiment..."
./run_tensorrt_llama_experiment.sh

echo "All experiments completed successfully." 