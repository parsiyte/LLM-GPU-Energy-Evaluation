#!/bin/bash
# Script to run targeted GPU profiling experiments for TensorRT models.

# Exit immediately if a command exits with a non-zero status.
set -e

# === Install yq if not installed ===
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && chmod +x /usr/bin/yq
fi

# === Hugging Face login using token ===
if [ -z "$HF_TOKEN" ]; then
  echo "Hugging Face token not provided! Please set HF_TOKEN environment variable."
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# === Initial DEPO setup ===
echo "Preparing DEPO and TensorRT..."
../prepare_tensorrt.sh

# Define the injection path dynamically
INJECTION_PATH="$(cd ../.. && pwd)/split/profiling_injection/libinjection_2.so"

# Base directory for all results
BASE_RESULTS_DIR="tensorrt_gpu_profiling_experiments"
mkdir -p "$BASE_RESULTS_DIR"

# Reboot GPU
echo "Rebooting GPU..."
nvidia-smi -r

# --- Function to run profiling for a given model ---
run_profiling() {
    local model_name=$1
    local gpu_id_smi=$2 # For nvidia-smi -i
    local cuda_device=$3 # For CUDA_VISIBLE_DEVICES
    local depo_gpu_id=$4
    local run_special_edp_case=$5

    echo "================================================="
    echo "Starting profiling for $model_name on GPU with CUDA_VISIBLE_DEVICES=$cuda_device"
    echo "================================================="

    export CUDA_VISIBLE_DEVICES=$cuda_device
    local model_script="../model_scripts/${model_name}.sh"

    if [ ! -x "$model_script" ]; then
      echo "Error: $model_script not found or not executable. Skipping."
      return
    fi
    # Prepare model
    echo "Preparing model: $model_name"
    "$model_script" > /dev/null 2>&1

    local model_results_dir="${BASE_RESULTS_DIR}/${model_name}_CUDA${cuda_device}"
    mkdir -p "$model_results_dir"

    # --- No-Tuning Profile (with DEPO) ---
    echo "--- Profiling No-Tuning for $model_name ---"
    local notuning_dir="${model_results_dir}/notuning_profile"
    mkdir -p "$notuning_dir"
    log_file="${notuning_dir}/${model_name}_notuning_gpu_log.csv"
    local output_file="${notuning_dir}/EP_stdout"
    
    rm -rf gpu_experiment_* kernels_count redirected.txt
    yq e -i ".msTestPhasePeriod = 6400" config.yaml
    yq e -i ".repeatTuningPeriodInSec = 0" config.yaml
    yq e -i ".doWaitPhase = 0" config.yaml
    yq e -i ".targetMetric = 0" config.yaml

    echo "Starting nvidia-smi for no-tuning run..."
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 500 -i $gpu_id_smi > "$log_file" &
    smi_pid=$!
    
    local start_time=$(date +%s)
    echo "Running DEPO --no-tuning..."
    CUDA_INJECTION64_PATH=$INJECTION_PATH \
    ../../split/build/apps/DEPO/DEPO --no-tuning --gpu $depo_gpu_id "$model_script" > "$output_file" 2>/dev/null
    local end_time=$(date +%s)
    kill $smi_pid
    
    local total_time=$((end_time - start_time))
    local periodic_time=$((total_time / 6))
    if [ "$periodic_time" -lt 1 ]; then periodic_time=1; fi
    echo "App time (no tuning) = $total_time sec -> periodic = $periodic_time sec"
    mv gpu_experiment_* kernels_count redirected.txt "$notuning_dir/" 2>/dev/null || true # Move any output files
    echo "--- Finished No-Tuning for $model_name ---"
    echo

    # --- EDP Periodic Wait Profile (Conditional) ---
    if [ "$run_special_edp_case" = true ]; then
        echo "--- Profiling EDP Periodic Wait for $model_name ---"
        local edp_dir="${model_results_dir}/edp_periodic_wait_profile"
        mkdir -p "$edp_dir"
        log_file="${edp_dir}/${model_name}_edp_periodic_wait_gpu_log.csv"
        
        rm -rf gpu_experiment_* kernels_count redirected.txt
        yq e -i ".targetMetric = 1" config.yaml # EDP
        yq e -i ".repeatTuningPeriodInSec = $periodic_time" config.yaml
        yq e -i ".doWaitPhase = 1" config.yaml # Wait phase ON

        echo "Starting nvidia-smi for EDP run..."
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 500 -i $gpu_id_smi > "$log_file" &
        smi_pid=$!

        echo "Running DEPO with EDP, periodic tuning, and wait phase..."
        CUDA_INJECTION64_PATH=$INJECTION_PATH \
        ../../split/build/apps/DEPO/DEPO --edp --gss --gpu $depo_gpu_id "$model_script" > "${edp_dir}/model_stdout.log" 2>&1

        kill $smi_pid
        mv gpu_experiment_* kernels_count redirected.txt "$edp_dir/" 2>/dev/null || true
        echo "--- Finished EDP Periodic Wait for $model_name ---"
        echo
    fi
}

# --- Define models and run experiments ---

# A100 Models (smi-id 1, cuda-dev 0, depo-gpu 1)
declare -A a100_models_edp
a100_models_edp["tensorrt_deepseek_32b"]=true
a100_models_edp["tensorrt_llama_3_1_8b"]=true

declare -a a100_models=(
    "tensorrt_deepseek_32b" "tensorrt_llama_3_1_8b" "tensorrt_cnn_llama_3_1_8b"
)
for model in "${a100_models[@]}"; do
    run_profiling "$model" 1 0 1 ${a100_models_edp[$model]:-false}
done

# A4500 Models (smi-id 0, cuda-dev 1, depo-gpu 0)
declare -A a4500_models_edp
a4500_models_edp["tensorrt_a4500_llama_3_1_8b"]=true

declare -a a4500_models=(
    "tensorrt_a4500_llama_3_1_8b" "tensorrt_cnn_a4500_llama_3_1_8b"
)
for model in "${a4500_models[@]}"; do
    run_profiling "$model" 0 1 0 ${a4500_models_edp[$model]:-false}
done

echo "All TensorRT profiling runs completed."
