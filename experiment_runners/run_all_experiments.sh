#!/bin/bash

# This script runs all experiments in a predefined order.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Run Pythia Experiments ---
echo "Running Pythia Experiments..."
./run_pythia_experiments.sh

# --- Run Llama Experiment (A100) ---
echo "Running Llama Experiment (A100)..."
./run_llama_experiment.sh

# --- Run Llama Experiment (A4500) ---
echo "Running Llama Experiment (A4500)..."
./run_llama_experiments_a4500.sh

# --- Run Quantized Llama Experiments (A4500) ---
echo "Running Quantized Llama Experiments (A4500)..."
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

# --- Run TensorRT Llama Experiment (A100) ---
echo "Running TensorRT Llama Experiment (A100)..."
./run_tensort_a100_llama_experiment.sh

# --- Run TensorRT Llama Experiment (A4500) ---
echo "Running TensorRT Llama Experiment (A4500)..."
./run_tensorrt_a4500_llama_experiment.sh

echo "All experiments completed successfully."
