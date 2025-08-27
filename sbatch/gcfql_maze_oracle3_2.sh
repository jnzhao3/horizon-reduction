#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --requeue
#SBATCH --array=1-1%16

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=2
JOB_N=2

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
source ~/.bashrc
micromamba activate aorl
export PYTHONPATH="../:${PYTHONPATH}"

declare -a commands=(
 [1]='python3 main.py --run_group gcfql_maze_oracle3_2 --env_name humanoidmaze-large-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-large-navigate-v0 --train_data_size 100000 --offline_steps 2000000 --save_interval 500000 --save_dir ../../scratch/gcfql/ --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=True --agent.subgoal_steps 25 --agent.value_loss_type squared --json_path ../jsons/data.json --agent.awr_invtemp 0.0'
 [2]='python3 main.py --run_group gcfql_maze_oracle3_2 --env_name humanoidmaze-large-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-large-navigate-v0 --train_data_size 100000 --offline_steps 2000000 --save_interval 500000 --save_dir ../../scratch/gcfql/ --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=True --agent.subgoal_steps 25 --agent.value_loss_type squared --json_path ../jsons/data.json --agent.awr_invtemp 1.0'
)

cd /home/jennifer/aorl/horizon_reduction

parallel --delay 20 --linebuffer -j 2 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
