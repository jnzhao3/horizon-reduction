from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'

CKPT_NUM = 1000000

##=========== IMPORTS ===========##

import argparse
import json
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
from agents import agents
from tqdm import tqdm
from utils.datasets import Dataset, GCDataset, HGCDataset, CGCDataset
from utils.flax_utils import restore_agent
from wrappers.datafuncs_utils import make_env_and_datasets


##=========== FLAGS ===========##

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--num_trial_steps', type=int, default=6000)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument(
    '--env_dataset_path',
    type=str,
    default='../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/humanoidmaze-giant-navigate-v0-010.npz',
)
parser.add_argument('--wandb_project', type=str, default='aorl2')
parser.add_argument('--wandb_entity', type=str, default='moma1234')
parser.add_argument('--wandb_group', type=str, default='2026-04-21-01')
parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))

args = vars(parser.parse_args())

run_name = f"dqc_goal_rollout_trials{args['num_trials']}_steps{args['num_trial_steps']}_seed{args['seed']}"
wandb_run = wandb.init(
    project=args['wandb_project'],
    entity=args['wandb_entity'],
    group=args['wandb_group'],
    name=run_name,
    mode=args['wandb_mode'],
    config={
        **args,
        'restore_path': PATH,
        'restore_checkpoint': CKPT_NUM,
    },
    settings=wandb.Settings(start_method='thread'),
)


def log_wandb(metrics, step=None):
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def make_rollout_figure(all_cell_points, replay_buffer, subgoals_buffer, start, goal, title):
    replay_buffer = np.asarray(replay_buffer)
    subgoals_buffer = np.asarray(subgoals_buffer)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x=all_cell_points[..., 0], y=all_cell_points[..., 1], s=10, alpha=0.1, c='gray')
    if replay_buffer.size > 0:
        ax.scatter(
            x=replay_buffer[..., 0],
            y=replay_buffer[..., 1],
            c=np.arange(len(replay_buffer)),
            cmap='viridis',
            s=1,
        )
    if subgoals_buffer.size > 0:
        ax.scatter(x=subgoals_buffer[..., 0], y=subgoals_buffer[..., 1], c='orange', s=8)
    ax.scatter(x=goal[0], y=goal[1], c='green', marker='*')
    ax.scatter(x=start[0], y=start[1], marker='x', c='red')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    return fig


def reset_to_task(env, task_info):
    reset_task_info = {
        'init_ij': task_info['init_ij'],
        'goal_ij': task_info['goal_ij'],
    }
    return env.reset(options=dict(task_info=reset_task_info))

##=========== MAIN ===========##

flags_path = os.path.join(PATH, 'flags.json')
with open(flags_path, 'r') as f:
    saved_flags = json.load(f)

wandb.run.config.update(
    {
        'restored_env_name': saved_flags.get('env_name'),
        'restored_agent': saved_flags.get('agent', {}),
    },
    allow_val_change=True,
)

agent_config = saved_flags['agent']

# %%
dataset_class_name = agent_config.get('dataset_class', 'GCDataset')
dataset_class = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'CGCDataset': CGCDataset,
}[dataset_class_name]

dataset_path = args['env_dataset_path']
if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f'Dataset path does not exist: {dataset_path}. '
        'Pass --env_dataset_path with the full .npz path.'
    )
dataset_npz = np.load(dataset_path)

fake_config = dict(
    env_name='humanoidmaze-giant-navigate-v0',
    # dataset_path='../../scratch/aorl2/YOUR_RUN_DIR/data-1000000.npz',
    dataset_path='../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/humanoidmaze-giant-navigate-v0-003.npz',
    observations_key='oracle_reps', # 'observations',
    goal_key='actor_goals',
    actions_key='low_actor_goals', #'actions',
    hidden_dims=(256, 256, 256),
    layer_norm=True,
    lr=3e-4,
    batch_size=256,
    num_train_steps=100000,
    log_interval=100,
    seed=0,
    value_p_curgoal=0.0,
    value_p_trajgoal=1.0,
    value_p_randomgoal=0.0,
    value_geom_sample=False,
    actor_p_curgoal=0.0,
    actor_p_trajgoal=1.0,
    actor_p_randomgoal=0.0,
    actor_geom_sample=True,
    gc_negative=False,
    subgoal_steps=25,
    discount=0.995,
    flow_steps=10,
    backup_horizon=25,
    goal_conditioned=False,
)

train_dataset = dataset_class(Dataset.create(**dict(dataset_npz)), config=fake_config)

# seed = saved_flags.get('seed', 0)
seed = args['seed']
example_batch = train_dataset.sample(1)

agent_config['train_goal_proposer'] = False

first_agent = agents[agent_config['agent_name']].create(seed, example_batch, agent_config)
first_agent = restore_agent(first_agent, PATH, CKPT_NUM)

print(f'Restored first_agent from checkpoint {CKPT_NUM}')

# %%
dqc_agent = first_agent

all_cells = {}

for ob in tqdm(train_dataset.dataset['observations']):
    key = (np.floor(ob[0]), np.floor(ob[1]))
    if key in all_cells:
        all_cells[key] += 1
    else:
        all_cells[key] = 1

all_cell_points = np.asarray(list(all_cells.keys()))
print(saved_flags['env_name'])

env, base_train_dataset, val_dataset = make_env_and_datasets(
    # saved_flags.get('env_name', 'humanoidmaze-giant-navigate-v0'),
    'humanoidmaze-giant-navigate-v0',
    dataset_path=args['env_dataset_path'],
    use_oracle_reps=True,
)

env.spec.max_episode_steps = args['num_trial_steps']

##=========== ROLLOUTS ===========##

replay_buffers = {}
subgoals_buffers = {}
successes = {}
rng = jax.random.PRNGKey(args['seed'])
task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

for cur_task_id, task_info in enumerate(task_infos, start=1):
    start = np.asarray(task_info['init_xy'])
    goal = np.asarray(task_info['goal_xy'])

    replay_buffers[f'task_{cur_task_id}'] = []
    subgoals_buffers[f'task_{cur_task_id}'] = []
    successes[f'task_{cur_task_id}'] = []

    print(f'running task {cur_task_id}')

    for trial in range(args['num_trials']):

        replay_buffer = []
        subgoals_buffer = []
        success = 0.0

        # ob, reset_info = reset_to_task(env, task_info)
        # rollout_goal = np.asarray(reset_info.get('goal', goal))[:2]
        ob, _ = env.reset(options=dict(task_id=cur_task_id))

        for s in tqdm(range(args['num_trial_steps'])):
            replay_buffer.append(ob)

            action_rng, rng = jax.random.split(rng)
            action = dqc_agent.sample_actions(observations=ob, goals=goal, seed=action_rng, best_of_n_override=2)
            action = np.asarray(np.clip(action, -1, 1))
            ob, reward, terminated, truncated, _ = env.step(action)

            if terminated or np.linalg.norm(ob[:2] - goal) < 0.05:
                print('finished')
                success = 1.0
                break

        replay_buffers[f'task_{cur_task_id}'].append(replay_buffer)
        subgoals_buffers[f'task_{cur_task_id}'].append(subgoals_buffer)
        successes[f'task_{cur_task_id}'].append(success)
        task_success_rate = float(np.mean(successes[f'task_{cur_task_id}']))
        completed_successes = [
            trial_success
            for task_successes in successes.values()
            for trial_success in task_successes
        ]
        overall_success_rate = float(np.mean(completed_successes))
        rollout_step = sum(len(v) for v in successes.values())

        fig = make_rollout_figure(
            all_cell_points,
            replay_buffer,
            subgoals_buffer,
            start,
            goal,
            (
                f'DQC direct-to-goal rollout, task {cur_task_id}, trial {trial + 1}, '
                f'success={success:.0f}'
            ),
        )
        log_wandb(
            {
                f'evaluation/task{cur_task_id}_success': success,
                f'evaluation/task{cur_task_id}_success_rate': task_success_rate,
                'evaluation/overall_success': overall_success_rate,
                'evaluation/completed_trials': sum(len(v) for v in successes.values()),
                f'evaluation/task{cur_task_id}_rollout': wandb.Image(fig),
            },
            step=rollout_step,
        )
        plt.close(fig)

final_success_metrics = {
    f'evaluation/final_task{task_id}_success_rate': float(np.mean(successes[f'task_{task_id}']))
    for task_id in range(1, len(task_infos) + 1)
}
final_success_metrics['evaluation/final_overall_success'] = float(
    np.mean([trial_success for task_successes in successes.values() for trial_success in task_successes])
)
log_wandb(final_success_metrics, step=sum(len(v) for v in successes.values()) + 1)
wandb.finish()
