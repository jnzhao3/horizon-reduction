#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --requeue
#SBATCH --array=1-2%100
#SBATCH --comment=2026-03-17-00-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=2

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-17-00_debug --offline_steps=50 --further_offline_steps=50 --collection_steps=50 --eval_episodes=1 --video_episodes=0 --eval_interval=5 --save_interval=25 --data_plot_interval=25 --log_interval=25 --cleanup=True --seed=1001 --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=4 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=300.0'
  [2]='MUJOCO_GL=egl python fql_baseline.py --run_group=2026-03-17-00_debug --offline_steps=50 --further_offline_steps=50 --collection_steps=50 --eval_episodes=1 --video_episodes=0 --eval_interval=5 --save_interval=25 --data_plot_interval=25 --log_interval=25 --cleanup=True --seed=1001 --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=agents/fql.py --agent.best_of_n=4 --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=300.0'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            