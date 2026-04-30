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
#SBATCH --qos=rail_gpu4_normal
#SBATCH --requeue
#SBATCH --array=1-2%100
#SBATCH --comment=2026-04-29-05-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=2

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python 19_data_collection_cube_safe.py --restore_path=../../scratch/2026-04-27-dqc/dqc/2026-04-27-dqc/sd200002s_33839966.0.33839966.2.20260427_210028/ --flow_restore_path=../../scratch/checkpoints/cube_quadruple_horizon_subgoal_proposer --flow_ckpt_num=1050000 --env_name=cube-quadruple-play-oraclerep-v0 --ckpt_num=1000000 --task_id=1 --subgoal_steps=250 --steps_to_subgoal=25 --num_additional_steps=50 --fql_train_steps=50 --fql_chunk_size=5 --fql_n_step=25 --fql_discount=0.999 --fql_alpha=300.0 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --dataset_replace_interval=1000 --seed=1000 --wandb_group=2026-04-29-05_debug --dataset_path=../../scratch/data/cube-quadruple-play-v0/cube-quadruple-play-v0.npz'
  [2]='MUJOCO_GL=egl python 19_data_collection_cube_safe.py --restore_path=../../scratch/2026-04-27-dqc/dqc/2026-04-27-dqc/sd200002s_33839966.0.33839966.2.20260427_210028/ --flow_restore_path=../../scratch/checkpoints/cube_quadruple_horizon_subgoal_proposer --flow_ckpt_num=1050000 --env_name=cube-quadruple-play-oraclerep-v0 --ckpt_num=1000000 --task_id=1 --subgoal_steps=250 --steps_to_subgoal=25 --num_additional_steps=50 --fql_train_steps=50 --fql_chunk_size=5 --fql_n_step=25 --fql_discount=0.999 --fql_alpha=300.0 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --dataset_replace_interval=10 --seed=1000 --wandb_group=2026-04-29-05_debug --dataset_dir=../../scratch/data/cube-quadruple-play-v0/'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            