#!/bin/bash

# This script runs all experiments four times and organizes the results.

# Exit immediately if a command exits with a non-zero status.
set -e

# Number of times to run all experiments
TOTAL_RUNS=4

# Base directory for all experiment results
BASE_RESULTS_DIR="all_model_experiments"

for i in $(seq 1 $TOTAL_RUNS); do
    echo "================================================="
    echo "Starting Experiment Run $i of $TOTAL_RUNS"
    echo "================================================="

    # Create a unique directory for this run
    RUN_DIR="${BASE_RESULTS_DIR}_$i"
    mkdir -p "$RUN_DIR"

    # --- Run Pythia Experiments ---
    echo "--- Running Pythia Experiments (Run $i) ---"
    ./run_pythia_experiments.sh
    # Consolidate Pythia results into the run directory
    mv all_pythia_run_experiments_results "$RUN_DIR/pythia_results"
    
    # --- Run Llama Experiment ---
    echo "--- Running Llama Experiment (Run $i) ---"
    ./run_llama_experiment.sh
    # Consolidate Llama results
    mv vllama_3_1_8b_results_a100 "$RUN_DIR/llama_results"
    
    # --- Run Llama Experiment on A4500 ---
    echo "--- Running Llama Experiment on A4500 (Run $i) ---"
    ./run_llama_experiments_a4500.sh
    # Consolidate Llama A4500 results
    mv vllama_3_1_8b_results_a4500 "$RUN_DIR/llama_a4500_results"

    # --- Run Quantized Llama Experiments ---
    echo "--- Running Quantized Llama Experiments (Run $i) ---"
    ./run_quantized_llama_experiments.sh
    # Consolidate Quantized Llama results
    mv quantized_llama_a4500_results "$RUN_DIR/quantized_llama_results"
    
    # --- Run Mistral Experiment on A100 ---
    echo "--- Running Mistral Experiment on A100 (Run $i) ---"
    ./run_mistrial_experiment_a100.sh
    # Consolidate Mistral results
    mv mistral_7b_a100_experiments "$RUN_DIR/mistral_results"
    
    # --- Run Deepseek 32B Experiment ---
    echo "--- Running Deepseek 32B Experiment (Run $i) ---"
    ./run_deepseek_32b_experiment.sh
    # Consolidate Deepseek results
    mv deepseek_32b_experiments_results "$RUN_DIR/deepseek_32b_results"
    
    # --- Run Qwen3 30B 3A Experiment ---
    echo "--- Running Qwen3 30B 3A Experiment (Run $i) ---"
    ./run_qwen3_30b_3a_experiment.sh
    # Consolidate Qwen3 results
    mv qwen3_30b_3a_experiments "$RUN_DIR/qwen3_results"

    # --- Run TensorRT Deepseek Experiments ---
    echo "--- Running TensorRT Deepseek Experiments (Run $i) ---"
    ./run_tensorrt_deepseek_experiments.sh
    # Consolidate TensorRT Deepseek results
    mv tensorrt_deepseek_32b_results "$RUN_DIR/tensorrt_deepseek_results"
    
    # --- Run TensorRT Llama Experiment ---
    echo "--- Running TensorRT Llama Experiment (Run $i) ---"
    ./run_tensorrt_a4500_llama_experiment.sh
    # Consolidate TensorRT Llama results
    mv llama_tensorrt_a4500_results "$RUN_DIR/tensorrt_llama_a4500_results"

    echo "--- Running TensorRT Llama Experiment on A100 (Run $i) ---"
    ./run_tensorrt_a100_llama_experiment.sh
    # Consolidate TensorRT Llama results
    mv llama_tensorrt_a100_results "$RUN_DIR/tensorrt_llama_a100_results"

    echo "================================================="
    echo "Finished Experiment Run $i. Results are in $RUN_DIR"
    echo "================================================="
    echo
done

echo "All $TOTAL_RUNS experiment runs completed successfully." 