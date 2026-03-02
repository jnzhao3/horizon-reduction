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
#SBATCH --comment=2026-02-27-00-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=6

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl
export PYTHONPATH="../:${PYTHONPATH}"


declare -a commands=(
  [1]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=1001 --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0'
  [2]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=1001 --env_name=antmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.99 --agent.batch_size=256 --agent.alpha=3.0'
  [3]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=2002 --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0'
  [4]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=2002 --env_name=antmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.99 --agent.batch_size=256 --agent.alpha=3.0'
  [5]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=3003 --env_name=humanoidmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/humanoidmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.995 --agent.batch_size=256 --agent.alpha=30.0'
  [6]='MUJOCO_GL=egl python e2e.py --run_group=2026-02-27-00 --offline_steps=200000 --collection_steps=1000000 --save_interval=100000 --eval_interval=50000 --seed=3003 --env_name=antmaze-medium-navigate-oraclerep-v0 --dataset_dir=../../scratch/data/antmaze-medium-navigate-v0 --agent=../agents/gcfql.py --wrapper=wrappers/rndsubgoals.py --wrapper.max_episode_steps=2000 --wrapper.pre_init=True --agent.actor_type=best-of-n --agent.train_goal_proposer=True --agent.goal_proposer_type=actor-gc --agent.num_qs=10 --agent.q_agg=mean --agent.discount=0.99 --agent.batch_size=256 --agent.alpha=3.0'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            