#!/bin/bash

# === Install yq if not installed ===
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq
    chmod +x /usr/bin/yq
fi

# === Hugging Face login using token ===
if [ -z "$HF_TOKEN" ]; then
  echo "Hugging Face token not provided! Please set HF_TOKEN environment variable."
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# === Initial DEPO setup ===
../prepare_depo.sh
pip install vllm==0.8.4
export CUDA_VISIBLE_DEVICES=0

# Define the injection path dynamically
INJECTION_PATH="$(cd ../.. && pwd)/split/profiling_injection/libinjection_2.so"

# Sampling config for Phase 2 baseline
SAMPLE_INTERVAL_MS=200
SAMPLE_DURATION_SEC=10
# Match the DEPO --gpu index used below
NVIDIA_SMI_GPU_INDEX=1

# === Reboot GPU ===
echo "Rebooting GPU..."
nvidia-smi -r

nvidia-smi -pm 1

# === Define models ===
models=(
  "v_mistral_7b"
)

# === Define metrics and flags ===
declare -A metrics
metrics[0]="--en"
metrics[1]="--edp"
metrics[2]="--eds"

# === Associative arrays to store timings per model ===
declare -A periodic_times
declare -A test_phase_periods

# === Phase 1: Prepare/Verify all models ===
echo "=== Phase 1: Preparing all models ==="
for model in "${models[@]}"; do
  echo "Preparing/Verifying model $model..."
  model_script="../model_scripts/${model}.sh"
  if [ ! -x "$model_script" ]; then
    echo "Error: $model_script not found or not executable. Skipping $model."
    continue
  fi
  "$model_script" > /dev/null 2>&1
done
echo "=== Phase 1: Finished preparing models ==="
echo

# === Phase 2: Run No-Tuning for all models and get timings ===
echo "=== Phase 2: Running No-Tuning and getting timings ==="
for model in "${models[@]}"; do
  echo "--- Running No-Tuning for $model ---"
  model_script="../model_scripts/${model}.sh"
  if [ ! -x "$model_script" ]; then
    echo "Skipping $model as script was not found/executable in Phase 1."
    continue
  fi

  none_tuning_dir="none_tuning_${model}"
  mkdir -p "$none_tuning_dir"
  output_file="${none_tuning_dir}/EP_stdout"
  smi_outfile="${none_tuning_dir}/${model}_smi_log.csv"

  echo "Cleaning up leftovers..."
  rm -rf gpu_experiment_* kernels_count redirected.txt

  echo "Setting default config for no-tuning run..."
  yq e -i ".msTestPhasePeriod = 6400" config.yaml
  yq e -i ".repeatTuningPeriodInSec = 0" config.yaml
  yq e -i ".doWaitPhase = 0" config.yaml
  yq e -i ".targetMetric = 0" config.yaml

  echo "Profiling application time for $model (no tuning)..."
  # Start nvidia-smi sampling in background
  timeout "${SAMPLE_DURATION_SEC}s" nvidia-smi -i "$NVIDIA_SMI_GPU_INDEX" \
    --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,nounits -lms "$SAMPLE_INTERVAL_MS" >> "$smi_outfile" 2>/dev/null &
  sampler_pid=$!
  
  START=$(date +%s)
  CUDA_INJECTION64_PATH=$INJECTION_PATH \
  ../../split/build/apps/DEPO/DEPO --no-tuning --gpu 1 "$model_script" > "$output_file" 2>/dev/null
  END=$(date +%s)
  # Ensure sampler finishes (or is already done)
  wait $sampler_pid 2>/dev/null || true
  total_time=$((END - START))
  periodic_time=$((total_time / 3))

  if [ "$total_time" -lt 3 ]; then
      echo "Warning: Total time ($total_time sec) is very short. Setting periodic time to 1 sec."
      periodic_time=1
  fi
  echo "App time (no tuning) = $total_time sec -> periodic = $periodic_time sec"

  if [ "$total_time" -gt 1000 ]; then
    test_phase_period=12800
    echo "Model time ($total_time sec) > 1000 sec. TestPhasePeriod will be $test_phase_period ms."
  else
    test_phase_period=6400
    echo "Model time ($total_time sec) <= 1000 sec. TestPhasePeriod will be $test_phase_period ms."
  fi

  periodic_times["$model"]=$periodic_time
  test_phase_periods["$model"]=$test_phase_period

  echo "Saving no-tuning results to $none_tuning_dir..."
  cp config.yaml "$none_tuning_dir/"
  if [ -f kernels_count ]; then
      cp kernels_count "$none_tuning_dir/"
      rm -f kernels_count
  else
      echo "Warning: kernels_count not found after no-tuning run for $model."
      touch "${none_tuning_dir}/kernels_count"
  fi
  if compgen -G "gpu_experiment_*" > /dev/null; then
      echo "Moving gpu_experiment_* output..."
      mv gpu_experiment_* "$none_tuning_dir/" 2>/dev/null
  else
      echo "No gpu_experiment_* output found for $model no-tuning run."
  fi
   if [ -f redirected.txt ]; then
       cp redirected.txt "$none_tuning_dir/"
       rm -f redirected.txt
   fi
  echo "--- Finished No-Tuning for $model ---"
  echo
done
echo "=== Phase 2: Finished No-Tuning runs ==="
echo

# === Phase 3: Run Experiments for all models ===
echo "=== Phase 3: Running Experiments ==="
for model in "${models[@]}"; do
  echo "--- Running Experiments for $model ---"
  model_script="../model_scripts/${model}.sh"
   if [ ! -x "$model_script" ]; then
    echo "Skipping $model as script was not found/executable in Phase 1."
    continue
  fi

  current_periodic_time=${periodic_times["$model"]}
  current_test_phase_period=${test_phase_periods["$model"]}

  if [ -z "$current_periodic_time" ] || [ -z "$current_test_phase_period" ]; then
      echo "Error: Timings not found for model $model. Skipping experiments."
      continue
  fi

  echo "Using Periodic Time: $current_periodic_time sec, Test Phase Period: $current_test_phase_period ms"
  yq e -i ".msTestPhasePeriod = $current_test_phase_period" config.yaml

  experiments_parent_dir="${model}_experiments"
  mkdir -p "$experiments_parent_dir"

  # Move the no-tuning results into the main experiments folder for this model
  none_tuning_dir="none_tuning_${model}"
  if [ -d "$none_tuning_dir" ]; then
    echo "Moving $none_tuning_dir into $experiments_parent_dir..."
    mv "$none_tuning_dir" "$experiments_parent_dir/"
  fi

  for metric_idx in 0 1 2; do
    for wait_phase in 0 1; do
      label=""
      if [ "$wait_phase" -eq 0 ]; then
        label="periodic_nowait"
      else
        label="periodic_wait"
      fi

      yq e -i ".targetMetric = $metric_idx" config.yaml
      yq e -i ".repeatTuningPeriodInSec = $current_periodic_time" config.yaml
      yq e -i ".doWaitPhase = $wait_phase" config.yaml

      folder_name="${model}_exp_${metrics[$metric_idx]##--}_$label"
      exp_folder_path="${experiments_parent_dir}/${folder_name}"
      mkdir -p "$exp_folder_path"

      echo " ^f^r Running experiment: $folder_name"
      rm -rf gpu_experiment_* kernels_count redirected.txt

      CUDA_INJECTION64_PATH=$INJECTION_PATH \
      ../../split/build/apps/DEPO/DEPO --gss --gpu 1 ${metrics[$metric_idx]} "$model_script"

      echo "Saving results to $exp_folder_path..."
      if compgen -G "gpu_experiment_*" > /dev/null; then
          mv gpu_experiment_* "$exp_folder_path/" 2>/dev/null
      else
          echo "Warning: No gpu_experiment_* output found for $folder_name"
      fi
      if [ -f redirected.txt ]; then
          cp redirected.txt "$exp_folder_path/"
          rm -f redirected.txt
      else
          echo "Warning: redirected.txt not found for $folder_name"
      fi
      if [ -f kernels_count ]; then
          cp kernels_count "$exp_folder_path/"
          rm -f kernels_count
      else
          echo "Warning: kernels_count not found for $folder_name"
          touch "${exp_folder_path}/kernels_count"
      fi
      cp config.yaml "$exp_folder_path/"
      echo
    done

    # Non-periodic experiment
    yq e -i ".targetMetric = $metric_idx" config.yaml
    yq e -i ".repeatTuningPeriodInSec = 0" config.yaml
    yq e -i ".doWaitPhase = 0" config.yaml

    folder_name="${model}_exp_${metrics[$metric_idx]##--}_none"
    exp_folder_path="${experiments_parent_dir}/${folder_name}"
    mkdir -p "$exp_folder_path"

    echo " ^f^r Running experiment: $folder_name"
    rm -rf gpu_experiment_* kernels_count redirected.txt

    CUDA_INJECTION64_PATH=$INJECTION_PATH \
    ../../split/build/apps/DEPO/DEPO --gss --gpu 1 ${metrics[$metric_idx]} "$model_script"

    echo "Saving results to $exp_folder_path..."
    if compgen -G "gpu_experiment_*" > /dev/null; then
        mv gpu_experiment_* "$exp_folder_path/" 2>/dev/null
    else
        echo "Warning: No gpu_experiment_* output found for $folder_name"
    fi
    if [ -f redirected.txt ]; then
        cp redirected.txt "$exp_folder_path/"
        rm -f redirected.txt
    else
        echo "Warning: redirected.txt not found for $folder_name"
    fi
    if [ -f kernels_count ]; then
        cp kernels_count "$exp_folder_path/"
        rm -f kernels_count
    else
        echo "Warning: kernels_count not found for $folder_name"
        touch "${exp_folder_path}/kernels_count"
    fi
    cp config.yaml "$exp_folder_path/"
    echo
  done
  echo "--- Finished Experiments for $model ---"
  echo
done

# === Phase 4: Consolidate Results ===
echo "=== Phase 4: Consolidating Results ==="
final_results_dir="mistral_7b_a100_experiments"
mkdir -p "$final_results_dir"

echo "Moving results to $final_results_dir..."
# Since the script is now focused on a single model, we can directly use the model name
# or iterate if we want to keep it flexible, but for this specific request, direct is fine.
# Assuming 'v_mistral_7b' is the only model processed as per current script setup.

model_name="v_mistral_7b" # Explicitly set for clarity, or could be derived if models array had more

experiments_parent_dir_to_move="${model_name}_experiments"

if [ -d "$experiments_parent_dir_to_move" ]; then
  echo "Moving $experiments_parent_dir_to_move to $final_results_dir/"
  mv "$experiments_parent_dir_to_move" "$final_results_dir/" || echo "Warning: Failed to move $experiments_parent_dir_to_move"
else
  echo "Warning: Directory $experiments_parent_dir_to_move not found."
fi

echo "=== All requested phases finished ==="
