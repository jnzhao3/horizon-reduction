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
#SBATCH --qos=rail_gpu4_normal
#SBATCH --requeue
#SBATCH --array=1-4%16

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=2
JOB_N=8

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
source ~/.bashrc
micromamba activate aorl
export PYTHONPATH="../:${PYTHONPATH}"

declare -a commands=(
 [1]='python3 e2e.py --run_group e2e_maze_3_2 --seed 0 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid sjs73s60'
 [2]='python3 e2e.py --run_group e2e_maze_3_2 --seed 0 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid wohn5op1'
 [3]='python3 e2e.py --run_group e2e_maze_3_2 --seed 1 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid f4tzqtdg'
 [4]='python3 e2e.py --run_group e2e_maze_3_2 --seed 1 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid dd5u9cyu'
 [5]='python3 e2e.py --run_group e2e_maze_3_2 --seed 2 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid 4m4wa4hx'
 [6]='python3 e2e.py --run_group e2e_maze_3_2 --seed 2 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid lyv0pf8t'
 [7]='python3 e2e.py --run_group e2e_maze_3_2 --seed 3 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid o3glpnyw'
 [8]='python3 e2e.py --run_group e2e_maze_3_2 --seed 3 --env_name humanoidmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/humanoidmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 1000 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/withrnd.py  --wbid a6wkcoaa'
)

cd /home/jennifer/aorl/horizon_reduction

parallel --delay 20 --linebuffer -j 2 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
