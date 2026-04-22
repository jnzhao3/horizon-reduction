#!/bin/bash

# List of scripts to run
scripts=(
  "MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=50 --num_trials=1 --num_trial_steps=50 --num_subgoals=16 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-20-00_debug"
)

# List of available GPU IDs (modify as needed)
gpus=(0 1 2 4 5 6 7)

num_gpus=${#gpus[@]}
num_scripts=${#scripts[@]}

# Store PIDs of background jobs
pids=()

# Function to handle Ctrl+C
cleanup() {
  echo "Terminating all running processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Function to run scripts sequentially on a given GPU
run_on_gpu() {
  local gpu_id=$1
  shift
  local gpu_scripts=("$@")

  for script in "${gpu_scripts[@]}"; do
    echo "Running $script on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id eval "$script" &
    pids+=($!)  # Store PID of the process
    wait ${pids[-1]}  # Wait for the process to finish before moving to the next
  done
}

# Distribute scripts among GPUs
for ((i=0; i<num_gpus; i++)); do
  gpu_scripts=()

  # Assign every nth script to this GPU
  for ((j=i; j<num_scripts; j+=num_gpus)); do
    gpu_scripts+=("${scripts[j]}")
  done

  if [ ${#gpu_scripts[@]} -gt 0 ]; then
    run_on_gpu ${gpus[i]} "${gpu_scripts[@]}" &
    pids+=($!)  # Store PID of background process
  fi
done

wait  # Wait for all background jobs
echo "All scripts finished."
