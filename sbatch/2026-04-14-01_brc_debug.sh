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
#SBATCH --array=1-1%100
#SBATCH --comment=2026-04-14-01-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=1

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python triangle_inequality.py --run_group=2026-04-14-01_debug --offline_steps=15 --further_offline_steps=50 --collection_steps=50 --eval_episodes=0 --video_episodes=0 --eval_interval=5 --save_interval=25 --data_plot_interval=25 --log_interval=25 --cleanup=True --seed=1001 --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --dataset_dir=../../scratch/data/humanoidmaze-large-navigate-v0 --agent=agents/dqc.py --agent.train_goal_proposer=True --agent.backup_horizon=25 --agent.policy_chunk_size=1 --agent.kappa_b=0.5 --agent.kappa_d=0.8 --agent.num_qs=2 --agent.q_agg=mean --agent.discount=0.999 --agent.batch_size=1028 --agent.best_of_n=32 --fql_agent=agents/fql.py --fql_agent.best_of_n=32 --fql_agent.horizon_length=25 --fql_agent.action_chunking=False --fql_agent.num_qs=10 --fql_agent.q_agg=mean --fql_agent.discount=0.995 --fql_agent.batch_size=256 --fql_agent.alpha=600.0 --triangle_percentile=50 --use_rnd_bonus=False'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            