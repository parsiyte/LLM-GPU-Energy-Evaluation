#!/bin/bash

# === Install yq if not installed ===
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq
    chmod +x /usr/bin/yq
fi

# === Initial environment setup ===
../prepare_depo.sh
pip install vllm==0.8.4
export CUDA_VISIBLE_DEVICES=0

# Define the injection path dynamically
INJECTION_PATH="$(cd ../.. && pwd)/split/profiling_injection/libinjection_2.so"

# === Reboot GPU ===
echo "Rebooting GPU..."
nvidia-smi -r

# === Define model ===
model="v_deepseek_32b"
model_script="../model_scripts/${model}.sh"

# === Define metrics and flags ===
declare -A metrics
metrics[0]="--en"
metrics[1]="--edp"
metrics[2]="--eds"

# === Phase 1: Prepare/Verify model ===
echo "=== Phase 1: Preparing model $model ==="
if [ ! -x "$model_script" ]; then
  echo "Error: $model_script not found or not executable."
  exit 1
fi
# Assuming the model script handles download/setup idempotently
"$model_script" > /dev/null 2>&1
echo "=== Phase 1: Finished preparing model ==="
echo

# === Phase 2: Run No-Tuning and get timings ===
echo "=== Phase 2: Running No-Tuning and getting timings for $model ==="
none_tuning_dir="none_tuning_${model}"
mkdir -p "$none_tuning_dir"
output_file="${none_tuning_dir}/EP_stdout"

# Clean up potential leftovers
rm -rf gpu_experiment_* kernels_count redirected.txt

echo "Setting default config for no-tuning run..."
yq e -i ".msTestPhasePeriod = 6400" config.yaml
yq e -i ".repeatTuningPeriodInSec = 0" config.yaml
yq e -i ".doWaitPhase = 0" config.yaml
yq e -i ".targetMetric = 0" config.yaml

echo "Profiling application time for $model (no tuning)..."
START=$(date +%s)
CUDA_INJECTION64_PATH=$INJECTION_PATH \
../../split/build/apps/DEPO/DEPO --no-tuning --gpu 1 "$model_script" > "$output_file" 2>/dev/null
END=$(date +%s)
total_time=$((END - START))
periodic_time=$((total_time / 6))

if [ "$total_time" -lt 3 ]; then
    echo "Warning: Total time ($total_time sec) is very short. Setting periodic time to 1 sec."
    periodic_time=1
fi
echo "App time (no tuning) = $total_time sec -> periodic = $periodic_time sec"

# Set TestPhasePeriod for this model's experiments
test_phase_period=12800
echo "TestPhasePeriod will be $test_phase_period ms."

# Save no-tuning results
echo "Saving no-tuning results to $none_tuning_dir..."
cp config.yaml "$none_tuning_dir/"
if [ -f kernels_count ]; then cp kernels_count "$none_tuning_dir/"; rm -f kernels_count; fi
if compgen -G "gpu_experiment_*" > /dev/null; then mv gpu_experiment_* "$none_tuning_dir/"; fi
if [ -f redirected.txt ]; then cp redirected.txt "$none_tuning_dir/"; rm -f redirected.txt; fi
echo "--- Finished No-Tuning for $model ---"
echo

# === Phase 3: Run Experiments ===
echo "=== Phase 3: Running Experiments for $model ==="
echo "Using Periodic Time: $periodic_time sec, Test Phase Period: $test_phase_period ms"

# Set the msTestPhasePeriod for this model's experiments
yq e -i ".msTestPhasePeriod = $test_phase_period" config.yaml

experiments_parent_dir="${model}_experiments"
mkdir -p "$experiments_parent_dir"

# Move the no-tuning results into the main experiments folder for this model
if [ -d "$none_tuning_dir" ]; then
  echo "Moving $none_tuning_dir into $experiments_parent_dir..."
  mv "$none_tuning_dir" "$experiments_parent_dir/"
fi

for metric in 0 1 2; do
  # Periodic experiments
  for wait_phase in 0 1; do
    if [ "$wait_phase" -eq 0 ]; then
      label="periodic_nowait"
    else
      label="periodic_wait"
    fi

    yq e -i ".targetMetric = $metric" config.yaml
    yq e -i ".repeatTuningPeriodInSec = $periodic_time" config.yaml
    yq e -i ".doWaitPhase = $wait_phase" config.yaml

    folder_name="exp_${metrics[$metric]##--}_$label"
    exp_folder_path="${experiments_parent_dir}/${folder_name}"
    mkdir -p "$exp_folder_path"

    echo "Running experiment: $folder_name"
    rm -rf gpu_experiment_* kernels_count redirected.txt
    
    CUDA_INJECTION64_PATH=$INJECTION_PATH \
    ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu 1 "$model_script"

    echo "Saving results to $exp_folder_path..."
    if compgen -G "gpu_experiment_*" > /dev/null; then mv gpu_experiment_* "$exp_folder_path/"; fi
    if [ -f redirected.txt ]; then cp redirected.txt "$exp_folder_path/"; rm -f redirected.txt; fi
    if [ -f kernels_count ]; then cp kernels_count "$exp_folder_path/"; rm -f kernels_count; fi
    cp config.yaml "$exp_folder_path/"
  done

  # Non-periodic experiment
  label="none"
  yq e -i ".targetMetric = $metric" config.yaml
  yq e -i ".repeatTuningPeriodInSec = 0" config.yaml
  yq e -i ".doWaitPhase = 0" config.yaml

  folder_name="exp_${metrics[$metric]##--}_$label"
  exp_folder_path="${experiments_parent_dir}/${folder_name}"
  mkdir -p "$exp_folder_path"

  echo "Running experiment: $folder_name"
  rm -rf gpu_experiment_* kernels_count redirected.txt
  
  CUDA_INJECTION64_PATH=$INJECTION_PATH \
  ../../split/build/apps/DEPO/DEPO ${metrics[$metric]} --gss --gpu 1 "$model_script"

  echo "Saving results to $exp_folder_path..."
  if compgen -G "gpu_experiment_*" > /dev/null; then mv gpu_experiment_* "$exp_folder_path/"; fi
  if [ -f redirected.txt ]; then cp redirected.txt "$exp_folder_path/"; rm -f redirected.txt; fi
  if [ -f kernels_count ]; then cp kernels_count "$exp_folder_path/"; rm -f kernels_count; fi
  cp config.yaml "$exp_folder_path/"
done
echo "--- Finished Experiments for $model ---"
echo

# === Phase 4: Consolidate Results ===
echo "=== Phase 4: Consolidating Results ==="
final_results_dir="deepseek_32b_experiments_results"
mkdir -p "$final_results_dir"

echo "Moving results to $final_results_dir..."
if [ -d "$experiments_parent_dir" ]; then mv "$experiments_parent_dir" "$final_results_dir/"; fi

# Reset msTestPhasePeriod in config
echo "Resetting msTestPhasePeriod to 6400"
yq e -i ".msTestPhasePeriod = 6400" config.yaml

echo "=== All phases finished ==="
