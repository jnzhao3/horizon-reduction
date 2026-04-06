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
#SBATCH --array=1-6%100
#SBATCH --comment=2026-04-06-01-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=6

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=1.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
  [2]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=1.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
  [3]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=3.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
  [4]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=3.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
  [5]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=10.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
  [6]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-06-01 --offline_steps=1000000 --further_offline_steps=2000000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/gcfql.py --agent.train_goal_proposer=True --agent.goal_proposer_type=default --agent.train_value=True --agent.horizon_length=25 --agent.action_chunking=False --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=10.0 --agent.subgoal_steps=25 --agent.awr_invtemp=0.0 --agent.best_of_n=4 --agent.use_policy_for_value=True --fql_agent=agents/fql.py --fql_agent.best_of_n=4 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            