#!/bin/bash

# === Install yq if not installed ===
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq
    chmod +x /usr/bin/yq
fi

# === Initial environment setup ===
../prepare_depo.sh
pip install vllm==0.9.2
export CUDA_VISIBLE_DEVICES=0


# Define the injection path dynamically
INJECTION_PATH="$(cd ../.. && pwd)/split/profiling_injection/libinjection_2.so"

# === Define metrics and corresponding flags ===
declare -A metrics
metrics[0]="--en"
metrics[1]="--edp"
metrics[2]="--eds"

# === Loop through combinations ===
for metric in 0 1 2; do
  for periodic in 0 250; do
    if [ "$periodic" -eq 0 ]; then
      # Only non-periodic, doWaitPhase doesn't matter here
      wait_phase=0
      label="none"
      
      # Update config
      yq e -i ".targetMetric = $metric" config.yaml
      yq e -i ".repeatTuningPeriodInSec = $periodic" config.yaml
      yq e -i ".doWaitPhase = $wait_phase" config.yaml

      # Run workload and DEPO
      CUDA_INJECTION64_PATH=$INJECTION_PATH \
      ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu 1 ../model_scripts/v_deepseek_32b.sh

      # Store results
      folder_name="exp_${metrics[$metric]##--}_${label}"
      mkdir "$folder_name"
      mv gpu_experiment_* "$folder_name"
      cp redirected.txt kernels_count config.yaml "$folder_name"

    else
      for wait_phase in 0 1; do
        if [ "$wait_phase" -eq 0 ]; then
          label="periodic_nowait"
        else
          label="periodic_wait"
        fi

        # Update config
        yq e -i ".targetMetric = $metric" config.yaml
        yq e -i ".repeatTuningPeriodInSec = $periodic" config.yaml
        yq e -i ".doWaitPhase = $wait_phase" config.yaml

        # Run workload and DEPO
        CUDA_INJECTION64_PATH=$INJECTION_PATH \
        ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu 1 ../model_scripts/v_deepseek_32b.sh

        # Store results
        folder_name="exp_${metrics[$metric]##--}_${label}"
        mkdir "$folder_name"
        mv gpu_experiment_* "$folder_name"
        cp redirected.txt kernels_count config.yaml "$folder_name"
      done
    fi
  done
done

# === Phase 4: Consolidate Results ===
echo "=== Phase 4: Consolidating Results ==="
final_results_dir="deepseek_32b_experiments_results"
mkdir -p "$final_results_dir"

echo "Moving results to $final_results_dir..."
if ls exp_* 1> /dev/null 2>&1; then
    mv exp_* "$final_results_dir/" || echo "Warning: Failed to move some experiment folders."
else
    echo "Warning: No 'exp_*' folders found to move."
fi

echo "=== All phases finished ==="

