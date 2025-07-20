#!/bin/bash

# This script runs all A4500-specific experiments four times and organizes the results.

# Exit immediately if a command exits with a non-zero status.
set -e

# Number of times to run all experiments
TOTAL_RUNS=4

# Base directory for all experiment results
BASE_RESULTS_DIR="a4500_model_experiments"

for i in $(seq 1 $TOTAL_RUNS); do
    echo "================================================="
    echo "Starting A4500 Experiment Run $i of $TOTAL_RUNS"
    echo "================================================="

    # Create a unique directory for this run
    RUN_DIR="${BASE_RESULTS_DIR}_$i"
    mkdir -p "$RUN_DIR"

    # --- Run Llama Experiment on A4500 ---
    echo "--- Running Llama Experiment on A4500 (Run $i) ---"
    ./run_llama_experiments_a4500.sh
    # Consolidate Llama A4500 results
    mv vllama_3_1_8b_results_a4500 "$RUN_DIR/llama_a4500_results"

    # --- Run TensorRT Llama Experiment ---
    echo "--- Running TensorRT Llama Experiment (Run $i) ---"
    ./run_tensorrt_llama_experiment.sh
    # Consolidate TensorRT Llama results
    mv llama_tensorrt_a4500_results "$RUN_DIR/tensorrt_llama_results"

    # --- Run Quantized Llama Experiments ---
    echo "--- Running Quantized Llama Experiments (Run $i) ---"
    ./run_quantized_llama_experiments.sh
    # Consolidate Quantized Llama results
    mv quantized_llama_a4500_results "$RUN_DIR/quantized_llama_results"
    
    echo "================================================="
    echo "Finished A4500 Experiment Run $i. Results are in $RUN_DIR"
    echo "================================================="
    echo
done

echo "All $TOTAL_RUNS A4500 experiment runs completed successfully." 