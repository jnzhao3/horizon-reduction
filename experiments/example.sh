#!/bin/bash

# List of scripts to run
scripts=(
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=6 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=32 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=7 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=6 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=5 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=8 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=32 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=32 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=4 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=3 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=3 --seed=3"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=512 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=2 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=16 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=4"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=0"
  "python train_parallel.py --wandb_tag=crawl_low_bs_250502 --agent.batch_size=8 --agent.updates_per_step=1 --benchmark=humanoid_bench --env_name=h1-crawl-v0 --agent.width_critic=256 --max_steps=1000000 --num_seeds=4 --seed=4"
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