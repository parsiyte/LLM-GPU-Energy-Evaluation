#!/bin/bash
# Script to run Llama-3.1-8B single and multi-GPU experiments.

# Exit immediately if a command exits with a non-zero status.
set -e
apt install -y curl

# === Ensure we have the correct yq (Mike Farah v4) ===
# Prefer system yq if it's Mike Farah; otherwise download a local copy and use it.
YQ_BIN="yq"
if command -v yq &> /dev/null; then
  if ! yq --version 2>/dev/null | grep -qi "mikefarah"; then
    echo "System yq is not Mike Farah's yq. Fetching local Mike Farah yq..."
    mkdir -p .tools
    curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o .tools/yq
    chmod +x .tools/yq
    YQ_BIN="$(pwd)/.tools/yq"
  fi
else
  echo "yq not found. Installing local Mike Farah yq..."
  mkdir -p .tools
  curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o .tools/yq
  chmod +x .tools/yq
  YQ_BIN="$(pwd)/.tools/yq"
fi

#=== Hugging Face login using token ===
if [ -z "$HF_TOKEN" ]; then
  echo "Hugging Face token not provided! Please set HF_TOKEN environment variable."
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# === Initial DEPO setup ===
echo "Preparing DEPO..."
../prepare_depo.sh

# === Activate Python virtual environment ===
echo "Activating virtual environment..."
source /data/lm-evaluation-harness/LLM-GPU-Energy-Evaluation/.venv/bin/activate

# Define the injection path dynamically
INJECTION_PATH="$(cd ../.. && pwd)/split/profiling_injection/libinjection_2.so"

# === Reboot GPU ===
echo "Rebooting GPU..."
nvidia-smi -r

# === Define metrics and flags ===
declare -A metrics
metrics[0]="--en"
metrics[1]="--edp"
metrics[2]="--eds"

# --- Function to collect experiment results ---
collect_results() {
    local exp_folder_path=$1
    local is_multigpu=$2

    echo "Collecting results..."
    mv gpu_experiment_* kernels_count redirected.txt "$exp_folder_path/" 2>/dev/null || true
    cp config.yaml "$exp_folder_path/"
}

# --- Function to run experiments for a given model ---
run_experiments() {
    local model_name=$1
    local depo_gpu_args=$2
    local test_phase_period=$3
    local periodic_time=$4
    local final_results_dir="${model_name}_results"
    local is_multigpu=false
    if [[ "$depo_gpu_args" == *,* ]]; then
        is_multigpu=true
    fi

    echo "================================================="
    echo "Starting experiments for $model_name"
    echo "DEPO GPU args: $depo_gpu_args"
    echo "Test Phase Period: $test_phase_period ms"
    echo "Periodic Time: $periodic_time sec"
    echo "================================================="

    # === Phase 1: Prepare model ===
    echo "--- Preparing model: $model_name ---"
    local model_script="../model_scripts/${model_name}.sh"
    if [ ! -x "$model_script" ]; then
      echo "Error: $model_script not found or not executable. Skipping."
      return
    fi
    "$model_script" > /dev/null 2>&1
    echo "--- Finished preparing model ---"
    echo

    # === Phase 2: Run No-Tuning to get timings ===
    echo "--- Running No-Tuning for $model_name to get execution time ---"
    local none_tuning_dir="none_tuning_${model_name}"
    mkdir -p "$none_tuning_dir"
    local output_file="${none_tuning_dir}/EP_stdout"

    rm -rf gpu_experiment_* kernels_count redirected.txt
    "$YQ_BIN" e -i ".msTestPhasePeriod = $test_phase_period" config.yaml
    "$YQ_BIN" e -i ".repeatTuningPeriodInSec = 0" config.yaml
    "$YQ_BIN" e -i ".doWaitPhase = 0" config.yaml
    "$YQ_BIN" e -i ".targetMetric = 0" config.yaml

    local start_time=$(date +%s)
    echo "Running DEPO --no-tuning..."
    CUDA_INJECTION64_PATH=$INJECTION_PATH \
    ../../split/build/apps/DEPO/DEPO --no-tuning --gpu $depo_gpu_args "$model_script" > "$output_file" 2>/dev/null
    local end_time=$(date +%s)
    
    local total_time=$((end_time - start_time))
    echo "App time (no tuning) = $total_time sec. Using fixed periodic time: $periodic_time sec"
    
    mv gpu_experiment_* kernels_count redirected.txt "$none_tuning_dir/" 2>/dev/null || true
    echo "--- Finished No-Tuning run for $model_name ---"
    echo

    # === Phase 3: Run Main Experiments ===
    echo "--- Running 9 main experiments for $model_name ---"
    local experiments_parent_dir="${model_name}_experiments_temp"
    mkdir -p "$experiments_parent_dir"
    mv "$none_tuning_dir" "$experiments_parent_dir/"

    for metric in 0 1 2; do
        # Periodic runs
        for wait_phase in 0 1; do
            local label=$([ "$wait_phase" -eq 0 ] && echo "periodic_nowait" || echo "periodic_wait")
            local folder_name="${model_name}_exp_${metrics[$metric]##--}_$label"
            local exp_folder_path="${experiments_parent_dir}/${folder_name}"
            mkdir -p "$exp_folder_path"
            
            echo "Running experiment: $folder_name"
            "$YQ_BIN" e -i ".targetMetric = $metric" config.yaml
            "$YQ_BIN" e -i ".repeatTuningPeriodInSec = $periodic_time" config.yaml
            "$YQ_BIN" e -i ".doWaitPhase = $wait_phase" config.yaml
            
            rm -rf gpu_experiment_*; rm -f kernels_count redirected.txt average_result.csv power_log.csv power_log.png power_log_gpu*.csv power_log_gpu*.png result.csv summed_results.csv
            CUDA_INJECTION64_PATH=$INJECTION_PATH \
            ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu $depo_gpu_args "$model_script" > "${exp_folder_path}/EP_stdout" 2>&1
            
            collect_results "$exp_folder_path" "$is_multigpu"
        done

        # Non-periodic (one-shot) run
        local folder_name="${model_name}_exp_${metrics[$metric]##--}_none"
        local exp_folder_path="${experiments_parent_dir}/${folder_name}"
        mkdir -p "$exp_folder_path"

        echo "Running experiment: $folder_name"
        "$YQ_BIN" e -i ".targetMetric = $metric" config.yaml
        "$YQ_BIN" e -i ".repeatTuningPeriodInSec = 0" config.yaml
        "$YQ_BIN" e -i ".doWaitPhase = 0" config.yaml

        rm -rf gpu_experiment_*; rm -f kernels_count redirected.txt average_result.csv power_log.csv power_log.png power_log_gpu*.csv power_log_gpu*.png result.csv summed_results.csv
        CUDA_INJECTION64_PATH=$INJECTION_PATH \
        ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu $depo_gpu_args "$model_script" > "${exp_folder_path}/EP_stdout" 2>&1
        
        collect_results "$exp_folder_path" "$is_multigpu"
    done
    echo "--- Finished main experiments for $model_name ---"
    echo

    # === Phase 4: Consolidate Results ===
    echo "--- Consolidating results for $model_name ---"
    local final_results_dir="${model_name}_results"
    mkdir -p "$final_results_dir"
    mv "$experiments_parent_dir"/* "$final_results_dir/"
    rm -r "$experiments_parent_dir"
    echo "Results for $model_name are in $final_results_dir"
    echo "================================================="
    echo
}

# --- Run Single-GPU Experiments ---
# msTestPhasePeriod=12400
run_experiments "vllama_3_1_8b" "0" 12400 200

# --- Run Multi-GPU Experiments ---
# msTestPhasePeriod=6400
run_experiments "vllama_3_1_8b_multi" "0,1" 6400 160

echo "All Llama-3.1-8B experiments completed."
