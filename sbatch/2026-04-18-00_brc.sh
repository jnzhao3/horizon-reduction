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
#SBATCH --array=1-54%10
#SBATCH --comment=2026-04-18-00-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=54

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [2]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [3]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [4]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [5]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [6]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [7]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [8]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [9]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [10]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [11]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [12]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [13]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [14]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [15]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [16]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=1001 --wandb_group=2026-04-18-00'
  [17]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=1001 --wandb_group=2026-04-18-00'
  [18]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=1001 --wandb_group=2026-04-18-00'
  [19]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [20]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [21]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [22]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [23]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [24]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [25]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [26]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [27]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [28]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [29]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [30]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [31]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [32]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [33]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [34]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=2002 --wandb_group=2026-04-18-00'
  [35]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=2002 --wandb_group=2026-04-18-00'
  [36]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=2002 --wandb_group=2026-04-18-00'
  [37]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [38]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [39]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
  [40]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [41]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [42]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.8 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
  [43]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [44]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [45]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
  [46]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [47]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [48]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=0.9 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
  [49]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [50]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [51]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=0.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
  [52]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.0 --seed=3003 --wandb_group=2026-04-18-00'
  [53]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=0.25 --seed=3003 --wandb_group=2026-04-18-00'
  [54]='MUJOCO_GL=egl python 07_tuning_goal_proposer.py --subgoal_steps=25 --steps_to_subgoal=25 --num_train_steps=100000 --num_trials=10 --num_trial_steps=2000 --num_subgoals=128 --mult_factor=1.0 --additive_factor=10.0 --A_B_factor=1.0 --B_C_factor=1.0 --seed=3003 --wandb_group=2026-04-18-00'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            