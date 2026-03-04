#!/bin/bash

# List of scripts to run
scripts=(
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=1001 --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=1 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0"
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=1001 --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=4 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=3.0"
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=2002 --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=1 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0"
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=2002 --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=4 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=3.0"
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=3003 --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=1 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0"
  "MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-03-00 --offline_steps=1000000 --further_offline_steps=1000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=3003 --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=4 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=3.0"
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
