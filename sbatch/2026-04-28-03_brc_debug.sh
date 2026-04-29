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
#SBATCH --comment=2026-04-28-03-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=1

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python 17_data_collection_giant.py --restore_path=../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/ --dataset_dir=../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/ --flow_restore_path=../../scratch/checkpoints/gc_flow_goal_proposer/observation_horizon_h1_100 --flow_ckpt_num=5000000 --env_name=humanoidmaze-giant-navigate-v0 --ckpt_num=1000000 --task_id=1 --subgoal_steps=100 --steps_to_subgoal=25 --num_additional_steps=50 --fql_train_steps=50 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1000 --wandb_group=2026-04-28-03_debug'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            