#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_low
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
 [1]='python3 e2e.py --run_group e2e_maze_4_4 --seed 0 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid 0wsazre8'
 [2]='python3 e2e.py --run_group e2e_maze_4_4 --seed 0 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid 0wfpi7v5'
 [3]='python3 e2e.py --run_group e2e_maze_4_4 --seed 1 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid mrqnczd4'
 [4]='python3 e2e.py --run_group e2e_maze_4_4 --seed 1 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid 8tfcmzyn'
 [5]='python3 e2e.py --run_group e2e_maze_4_4 --seed 2 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid wa2gj88g'
 [6]='python3 e2e.py --run_group e2e_maze_4_4 --seed 2 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid r60l5g7g'
 [7]='python3 e2e.py --run_group e2e_maze_4_4 --seed 3 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 100000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid eimjgzew'
 [8]='python3 e2e.py --run_group e2e_maze_4_4 --seed 3 --env_name antmaze-medium-navigate-oraclerep-v0 --agent ../agents/gcfql.py --dataset_dir ../../scratch/data/antmaze-medium-navigate-v0 --train_data_size 1000000 --save_dir ../../scratch --offline_steps 100 --save_interval 50 --log_interval 100 --eval_interval 50 --eval_episodes 0 --video_episodes 0 --agent.alpha 300 --agent.actor_type best-of-n --agent.train_goal_proposer=False --agent.actor_hidden_dims 512,512,512,512 --agent.value_hidden_dims 512,512,512,512 --agent.batch_size 256 --agent.num_actions 8 --agent.num_qs 10 --agent.q_agg mean --agent.discount 0.995 --data_option datafuncs/ogbench.py --data_option.collection_steps 100 --data_option.save_data_interval 50 --data_option.plot_interval 50 --data_option.max_episode_steps 20  --wbid 0se0q36g'
)

cd /home/jennifer/aorl/horizon_reduction

parallel --delay 20 --linebuffer -j 2 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
