#!/bin/bash

# This script runs all TensorRT experiments four times and organizes the results.

# Exit immediately if a command exits with a non-zero status.
set -e

# Number of times to run all experiments
TOTAL_RUNS=4

# Base directory for all experiment results
BASE_RESULTS_DIR="all_tensorrt_experiments"

for i in $(seq 1 $TOTAL_RUNS); do
    echo "================================================="
    echo "Starting TensorRT Experiment Run $i of $TOTAL_RUNS"
    echo "================================================="

    # Create a unique directory for this run
    RUN_DIR="${BASE_RESULTS_DIR}_$i"
    mkdir -p "$RUN_DIR"

    # --- Run TensorRT Deepseek Experiments ---
    echo "--- Running TensorRT Deepseek Experiments (Run $i) ---"
    ./run_tensorrt_deepseek_experiments.sh
    # Consolidate TensorRT Deepseek results
    mv tensorrt_deepseek_32b_results "$RUN_DIR/tensorrt_deepseek_results"
    
    # --- Run TensorRT Llama Experiment (A4500) ---
    echo "--- Running TensorRT Llama Experiment (A4500) (Run $i) ---"
    ./run_tensorrt_a4500_llama_experiment.sh
    # Consolidate TensorRT Llama results
    mv llama_tensorrt_a4500_results "$RUN_DIR/tensorrt_llama_a4500_results"

    # --- Run TensorRT Llama Experiment on A100 ---
    echo "--- Running TensorRT Llama Experiment on A100 (Run $i) ---"
    ./run_tensort_a100_llama_experiment.sh
    # Consolidate TensorRT Llama results
    mv llama_tensorrt_a100_results "$RUN_DIR/tensorrt_llama_a100_results"
    
    # --- Run TensorRT CNN Llama Experiment on A4500 ---
    echo "--- Running TensorRT CNN Llama Experiment on A4500 (Run $i) ---"
    ./run_tensorrt_a4500_cnn_experiment.sh
    # Consolidate TensorRT CNN Llama results
    mv cnn_tensorrt_a4500_results "$RUN_DIR/tensorrt_cnn_a4500_results"

    # --- Run TensorRT CNN Llama Experiment on A100 ---
    echo "--- Running TensorRT CNN Llama Experiment on A100 (Run $i) ---"
    ./run_tensorrt_a100_cnn_experiment.sh
    # Consolidate TensorRT CNN Llama results
    mv cnn_tensorrt_a100_results "$RUN_DIR/tensorrt_cnn_a100_results"

    echo "================================================="
    echo "Finished TensorRT Experiment Run $i. Results are in $RUN_DIR"
    echo "================================================="
    echo
done

echo "All $TOTAL_RUNS TensorRT experiment runs completed successfully."
