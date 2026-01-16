#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --requeue
#SBATCH --array=1-1%16
#SBATCH --signal=B:USR1@90

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=2
JOB_N=1

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
source ~/.bashrc
micromamba activate aorl
export PYTHONPATH="../:${PYTHONPATH}"

declare -a commands=(
 [1]='python3 simulation_3.py --task_start "(0.0, 0.0)" --task_end "(0.0, 1.0)" --waypoint "(0.0, 1.0)" --collection_steps 100 --train_steps 100 --eval_interval 50'
)

cd /home/jennifer/aorl/horizon_reduction

parallel --delay 20 --linebuffer -j 2 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
