from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'

DATASET_DIR = '../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/'

DATASET_PATH = DATASET_DIR + 'humanoidmaze-giant-navigate-v0-000.npz'

CKPT_NUM = 1000000

##=========== IMPORTS ===========##

import glob
import json
import os
import pathlib
import time

import numpy as np
from tqdm import tqdm
from agents import agents
from agents.fql import get_config as get_fql_config
from agents.goal_proposer import GCFlowGoalProposerAgent
from utils.datasets import Dataset, GCDataset, HGCDataset, CGCDataset, ReplayBuffer
from wrappers.datafuncs_utils import make_env_and_datasets, to_oracle_reps
from utils.evaluation import evaluate
from utils.statistics import get_statistics_class
from utils.networks import ActorVectorField

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP

from utils.flax_utils import restore_agent, save_agent
import argparse


##=========== FLAGS ===========##

parser = argparse.ArgumentParser()
parser.add_argument('--subgoal_steps', type=int, default=100)
parser.add_argument('--steps_to_subgoal', type=int, default=25)
parser.add_argument('--num_train_steps', type=int, default=3000000)
parser.add_argument('--num_additional_steps', type=int, default=1000000)
parser.add_argument('--fql_train_steps', type=int, default=1000000)
parser.add_argument('--fql_n_step', type=int, default=1)
parser.add_argument('--fql_discount', type=float, default=None, help='FQL discount; defaults to agent config discount if not set.')
parser.add_argument('--fql_alpha', type=float, default=None, help='FQL BC coefficient; defaults to agent config alpha if not set.')
parser.add_argument('--fql_log_interval', type=int, default=1000)
parser.add_argument('--fql_save_interval', type=int, default=100000)
parser.add_argument('--fql_eval_interval', type=int, default=50000)
parser.add_argument('--fql_eval_episodes', type=int, default=10)
parser.add_argument('--data_log_interval', type=int, default=1000)
parser.add_argument('--task_id', '--single_task_id', dest='task_id', type=int, default=1, help='1-indexed task id to collect data for.')
parser.add_argument('--num_subgoals', type=int, default=128)
parser.add_argument('--mult_factor', type=float, default=0.9)
parser.add_argument('--additive_factor', type=float, default=0.0)
parser.add_argument('--A_B_factor', type=float, default=1.0)
parser.add_argument('--B_C_factor', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--wandb_project', type=str, default='aorl2')
parser.add_argument('--wandb_entity', type=str, default='moma1234')
parser.add_argument('--wandb_group', type=str, default='giant_data_collection')
parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))
parser.add_argument('--save_dir', type=str, default='../../scratch/aorl2')
parser.add_argument('--replay_buffer_name', type=str, default=None)
parser.add_argument('--restore_path', type=str, default=PATH)
parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
parser.add_argument('--dataset_replace_interval', type=int, default=1000)
parser.add_argument('--num_datasets', type=int, default=None)
parser.add_argument('--ckpt_num', type=int, default=CKPT_NUM)
parser.add_argument('--env_name', type=str, default='humanoidmaze-giant-navigate-v0')
parser.add_argument('--flow_restore_path', type=str, default='../../scratch/checkpoints/gc_flow_goal_proposer/observation_horizon_h1_100')
parser.add_argument('--flow_ckpt_num', type=int, default=5000000)

args = vars(parser.parse_args())

if args['dataset_dir'] is None:
    datasets = [args['dataset_path']]
elif args['dataset_dir'].endswith('.npz'):
    datasets = [args['dataset_dir']]
else:
    datasets = [f for f in sorted(glob.glob(f'{args["dataset_dir"]}/*.npz')) if '-val.npz' not in f]
if args['num_datasets'] is not None:
    datasets = datasets[:args['num_datasets']]
dataset_idx = 0

run_name = (
    f"giant_goal_proposer_sg{args['subgoal_steps']}_train{args['num_train_steps']}_"
    f"mf{args['mult_factor']}_af{args['additive_factor']}_seed{args['seed']}"
)
wandb_run = wandb.init(
    project=args['wandb_project'],
    entity=args['wandb_entity'],
    group=args['wandb_group'],
    name=run_name,
    mode=args['wandb_mode'],
    config={
        **args,
        'restore_path': args['restore_path'],
        'restore_checkpoint': args['ckpt_num'],
    },
    dir='../../scratch/wandb',
    settings=wandb.Settings(start_method='thread'),
)

wandb.alert(
    title='Data collection run started',
    text=f'Run "{run_name}" has started.',
    level=wandb.AlertLevel.INFO,
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

##=========== MAIN ===========##

flags_path = os.path.join(args['restore_path'], 'flags.json')
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
agent_config['subgoal_steps'] = args['subgoal_steps']

dataset_class_name = agent_config.get('dataset_class', 'GCDataset')
dataset_class = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'CGCDataset': CGCDataset,
}[dataset_class_name]

# add placeholder flags
agent_config['actor_p_curgoal'] = 0.5
agent_config['actor_p_trajgoal'] = 0.5
agent_config['actor_p_randomgoal'] = 0.0
agent_config['actor_geom_sample'] = 0.5
agent_config['subgoal_steps'] = 100
agent_config['train_goal_proposer'] = False

dataset_npz = np.load(datasets[0])
train_dataset = dataset_class(Dataset.create(**dict(dataset_npz)), config=agent_config)

seed = args['seed']
example_batch = train_dataset.sample(1)

first_agent = agents[agent_config['agent_name']].create(seed, example_batch, agent_config)
first_agent = restore_agent(first_agent, args['restore_path'], args['ckpt_num'])

print(f'Restored first_agent from checkpoint {args["ckpt_num"]}')

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

config = dict(
    env_name=args['env_name'],
    dataset_path=datasets[0],
    observations_key='oracle_reps',
    goal_key='actor_goals',
    actions_key='low_actor_goals',
    hidden_dims=(256, 256, 256),
    layer_norm=True,
    lr=3e-4,
    batch_size=256,
    num_train_steps=args['num_train_steps'],
    log_interval=100,
    seed=args['seed'],
    value_p_curgoal=0.0,
    value_p_trajgoal=1.0,
    value_p_randomgoal=0.0,
    value_geom_sample=False,
    actor_p_curgoal=0.0,
    actor_p_trajgoal=1.0,
    actor_p_randomgoal=0.0,
    actor_geom_sample=True,
    gc_negative=False,
    subgoal_steps=args['subgoal_steps'],
    discount=0.999,
    flow_steps=10,
    backup_horizon=25,
    goal_conditioned=False,
    horizon_conditioned=True
)

env, base_train_dataset, val_dataset = make_env_and_datasets(
    config['env_name'],
    dataset_path=datasets[0],
    use_oracle_reps=True,
)
proposer_train_dataset = CGCDataset(base_train_dataset, config=config)


def _dataset_mapping(dataset):
    if hasattr(dataset, 'dataset'):
        return dataset.dataset
    if hasattr(dataset, '_dict'):
        return dataset._dict
    return dataset


def _prepare_initial_replay_data(dataset):
    data = {key: np.asarray(value).copy() for key, value in dataset.items()}
    size = len(data['observations'])

    if 'next_observations' not in data:
        next_observations = np.empty_like(data['observations'])
        next_observations[:-1] = data['observations'][1:]
        next_observations[-1] = data['observations'][-1]
        data['next_observations'] = next_observations

    terminals = data.get('terminals', np.zeros(size, dtype=np.float32)).astype(np.float32)
    data['terminals'] = terminals

    if 'rewards' not in data:
        data['rewards'] = np.zeros(size, dtype=np.float32)
    if 'masks' not in data:
        data['masks'] = 1.0 - terminals

    return data


def _build_fql_config(config, n_step=1, discount_override=None, alpha_override=None):
    fql_config = get_fql_config()
    fql_config['agent_name'] = 'fql'
    fql_config['batch_size'] = int(config['batch_size'])
    fql_config['discount'] = float(discount_override if discount_override is not None else config['discount'])
    fql_config['flow_steps'] = int(config['flow_steps'])
    fql_config['horizon_length'] = 1
    fql_config['action_chunking'] = False
    fql_config['encoder'] = None
    fql_config['n_step'] = int(n_step)
    if alpha_override is not None:
        fql_config['alpha'] = float(alpha_override)

    return fql_config


def _sample_replay_batch(buffer, batch_size):
    idxs = np.random.randint(buffer.size, size=batch_size)
    return buffer.sample(batch_size, idxs=idxs)


def _n_step_from_dict(data, batch_size, n_step, discount):
    """Compute n-step returns from a plain dict of arrays."""
    size = len(data['observations'])
    idxs = np.random.randint(size, size=batch_size)

    rewards = data['rewards'][idxs].copy().astype(np.float64)
    next_obs = data['next_observations'][idxs].copy().astype(np.float64)
    cumulative_mask = data['masks'][idxs].copy().astype(np.float64)

    for i in range(1, n_step):
        step_idxs = np.minimum(idxs + i, size - 1)
        rewards += (discount ** i) * cumulative_mask * data['rewards'][step_idxs]
        alive = cumulative_mask[:, None]
        next_obs = data['next_observations'][step_idxs] * alive + next_obs * (1.0 - alive)
        cumulative_mask = cumulative_mask * data['masks'][step_idxs]

    result = {k: v[idxs].copy() for k, v in data.items()}
    result['rewards'] = rewards.astype(data['rewards'].dtype)
    result['next_observations'] = next_obs.astype(data['next_observations'].dtype)
    result['masks'] = cumulative_mask.astype(data['masks'].dtype)
    return result


def _proportional_sample(base_data, new_buffer, total_base_size, batch_size, n_step=1, discount=0.99):
    R = new_buffer.size
    total = total_base_size + R
    n_new = int(np.random.binomial(batch_size, R / total)) if R > 0 else 0
    n_base = batch_size - n_new

    parts = []
    if n_base > 0:
        if n_step > 1:
            parts.append(_n_step_from_dict(base_data, n_base, n_step, discount))
        else:
            idxs = np.random.randint(len(base_data['observations']), size=n_base)
            parts.append({k: v[idxs] for k, v in base_data.items()})
    if n_new > 0:
        if n_step > 1:
            idxs = np.random.randint(R, size=n_new)
            parts.append(new_buffer.sample_n_step(n_new, n_step, discount, idxs=idxs))
        else:
            idxs = np.random.randint(R, size=n_new)
            parts.append(new_buffer.sample(n_new, idxs=idxs))

    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def _make_transition_for_buffer(buffer, ob, action, reward, done, truncated, next_ob, info, env):
    transition = {}
    oracle_rep = np.asarray(to_oracle_reps(obs=np.asarray(ob)[None], env=env))[0]

    explicit_values = {
        'observations': ob,
        'actions': action,
        'rewards': reward,
        'terminals': float(done),
        'truncates': float(truncated),
        'truncated': float(truncated),
        'masks': 1.0 - float(done),
        'valids': 1.0 - float(done),
        'next_observations': next_ob,
        'qpos': info.get('qpos'),
        'qvel': info.get('qvel'),
        'oracle_reps': oracle_rep,
    }

    for key, storage in buffer.items():
        value = explicit_values.get(key)
        if value is None:
            value = np.zeros(storage.shape[1:], dtype=storage.dtype)
        transition[key] = np.asarray(value, dtype=storage.dtype)
    return transition


def _save_replay_buffer(buffer, save_dir, name):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / name
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')

    data = {key: np.asarray(value[:buffer.size]) for key, value in buffer.items()}
    np.savez(path, **data)
    (save_dir / 'rbsize').write_text(str(buffer.max_size))
    (save_dir / 'replaybuffer_pointer').write_text(str(buffer.pointer))

    metadata_path = path.with_suffix('.json')
    metadata = {
        'size': int(buffer.size),
        'pointer': int(buffer.pointer),
        'max_size': int(buffer.max_size),
        'task_id': int(args['task_id']),
        'num_additional_steps': int(args['num_additional_steps']),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return path

##=========== RESTORE GOAL PROPOSER ===========##

example_batch = proposer_train_dataset.sample(1)
flow_agent = GCFlowGoalProposerAgent.create(example_batch, config)
flow_agent = restore_agent(flow_agent, args['flow_restore_path'], args['flow_ckpt_num'])
print(f'Restored flow_agent from {args["flow_restore_path"]} checkpoint {args["flow_ckpt_num"]}')

##=========== UTILITIES ===========##

# %%
def sigmoid(x):
    x = np.asarray(x)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )

def value_xy(ob, subgoals, goal, agent=dqc_agent):
    assert subgoals.ndim > 1, 'did not provide multiple subgoals'
    subgoal_obs = np.repeat(ob[None], len(subgoals), axis=0)
    subgoal_obs[..., :2] = subgoals
    goals = np.repeat(goal[None], len(subgoals), axis=0)

    vs = agent.network.select('value')(subgoal_obs, goals)
    return vs

def dynamical_distance(ob, subgoals, goal, agent=dqc_agent):

    all_obs = np.repeat(ob[None], len(subgoals), axis=0)
    ob_to_subgoal_vs = agent.network.select('value')(all_obs, subgoals)
    subgoal_to_goal_vs = value_xy(ob, subgoals, goal)
    ob_to_goal_v = agent.network.select('value')(ob, goal)

    ob_to_subgoal_vs = sigmoid(ob_to_subgoal_vs)
    subgoal_to_goal_vs = sigmoid(subgoal_to_goal_vs)
    ob_to_goal_v = sigmoid(ob_to_goal_v)

    gamma_to_subgoal = np.log(np.clip(ob_to_subgoal_vs, 1e-6, 1.0)) / np.log(dqc_agent.config['discount'])
    gamma_to_goal = np.log(np.clip(subgoal_to_goal_vs, 1e-6, 1.0)) / np.log(dqc_agent.config['discount'])
    ob_to_goal = np.log(np.clip(ob_to_goal_v, 1e-6, 1.0)) / np.log(dqc_agent.config['discount'])

    return gamma_to_subgoal, gamma_to_goal, ob_to_goal

def sample_n(ob, n, sample_rng, goal=None, agent=flow_agent):
    oracle_rep = np.asarray(to_oracle_reps(obs=np.asarray(ob)[None], env=env))[0]
    obs = np.repeat(oracle_rep[None], n, axis=0)
    if goal is not None:
        goals = np.repeat(goal[None], n, axis=0)
    else:
        goals = goal

    return flow_agent.sample_actions(obs, goals, sample_rng)

##=========== ROLLOUTS ===========##

subgoals_buffers = {}
successes = {}
rng = jax.random.PRNGKey(args['seed'])
task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

if args['task_id'] < 1 or args['task_id'] > len(task_infos):
    raise ValueError(f'--task_id must be in [1, {len(task_infos)}], got {args["task_id"]}')

cur_task_id = args['task_id']
task_info = task_infos[cur_task_id - 1]
start = np.asarray(task_info['init_xy'])
goal = np.asarray(task_info['goal_xy'])
task_key = f'task_{cur_task_id}'

_, base_ds, _ = make_env_and_datasets(
    config['env_name'],
    dataset_path=datasets[dataset_idx],
    use_oracle_reps=True,
)
current_base_data = _prepare_initial_replay_data(_dataset_mapping(base_ds))
single_dataset_size = len(current_base_data['observations'])
total_base_size = single_dataset_size * len(datasets)

example_transition = {k: v[0] for k, v in current_base_data.items()}
new_replay_buffer = ReplayBuffer.create(example_transition, args['num_additional_steps'])

subgoals_buffers[task_key] = []
successes[task_key] = []

print(f'running task {cur_task_id}')
print(f'loaded shard {dataset_idx}/{len(datasets)}: {single_dataset_size} transitions/shard, {total_base_size} total base; new buffer capacity={args["num_additional_steps"]}')

num_collected_steps = 0
num_trajectories = 0
num_subgoal_selections = 0
num_random_subgoal_selections = 0
random_subgoal_frac_ema = None
ema_alpha = 0.1
collection_stats = get_statistics_class(config['env_name'])(env=env)

with tqdm(total=args['num_additional_steps']) as pbar:
    ob, _ = env.reset(options=dict(task_id=cur_task_id))
    rollout_observations = []
    subgoals_buffer = []
    subgoal = None
    to_subgoal = 0
    last_mask_size = 0

    while num_collected_steps < args['num_additional_steps']:
        rollout_observations.append(ob.copy())

        if subgoal is None or to_subgoal == args['steps_to_subgoal']:
            rng, sample_rng = jax.random.split(rng)
            subgoals = np.asarray(sample_n(ob, args['num_subgoals'], sample_rng))

            gamma_to_subgoal, gamma_to_goal, ob_to_goal = dynamical_distance(ob, subgoals, goal)
            gamma_to_subgoal = np.asarray(gamma_to_subgoal).reshape(-1)
            gamma_to_goal = np.asarray(gamma_to_goal).reshape(-1)
            ob_to_goal = float(np.asarray(ob_to_goal).reshape(-1)[0])

            mask = gamma_to_goal < ob_to_goal * args['mult_factor'] + args['additive_factor']
            last_mask_size = int(np.sum(mask))
            num_subgoal_selections += 1
            if not np.any(mask):
                # print(f'no improving subgoal found after {num_collected_steps} additional steps')
                # break
                rng, key = jax.random.split(rng)
                subgoal = subgoals[jax.random.randint(key, (), 0, len(subgoals))]
                num_random_subgoal_selections += 1

            else:

                filtered_subgoals = subgoals[mask]
                filtered_scores = (args['A_B_factor'] * gamma_to_subgoal + args['B_C_factor'] * gamma_to_goal)[mask]

                subgoal = filtered_subgoals[int(np.argmin(filtered_scores))]

            subgoals_buffer.append(subgoal)
            to_subgoal = 0

        action_rng, rng = jax.random.split(rng)
        action = dqc_agent.sample_actions(observations=ob, goals=subgoal, seed=action_rng, best_of_n_override=2)
        action = np.asarray(np.clip(action, -1, 1))
        to_subgoal += 1
        next_ob, reward, terminated, truncated, info = env.step(action)
        num_collected_steps += 1
        budget_exhausted = num_collected_steps >= args['num_additional_steps']
        success = float(terminated or np.linalg.norm(next_ob[:2] - goal) < 0.05)
        budget_truncated = budget_exhausted and not (terminated or truncated or bool(success))
        done = terminated or truncated or budget_truncated or bool(success)

        transition = _make_transition_for_buffer(
            new_replay_buffer,
            ob=ob,
            action=action,
            reward=reward,
            done=done,
            truncated=truncated or budget_truncated,
            next_ob=next_ob,
            info=info,
            env=env,
        )
        new_replay_buffer.add_transition(transition)
        collection_stats.log_episode(ob, action)
        pbar.update(1)

        if args['data_log_interval'] > 0 and num_collected_steps % args['data_log_interval'] == 0:
            current_frac = num_random_subgoal_selections / num_subgoal_selections
            if random_subgoal_frac_ema is None:
                random_subgoal_frac_ema = current_frac
            else:
                random_subgoal_frac_ema = ema_alpha * current_frac + (1 - ema_alpha) * random_subgoal_frac_ema
            log_wandb(
                {
                    **{f'data_collection/{k}': v for k, v in collection_stats.get_statistics().items()},
                    'data_collection/random_subgoal_frac': current_frac,
                    'data_collection/random_subgoal_frac_ema': random_subgoal_frac_ema,
                    'data_collection/mask_size': last_mask_size,
                },
                step=config['num_train_steps'] + num_collected_steps,
            )

        if np.linalg.norm(next_ob[:2] - subgoal) < 0.1:
            subgoal = None

        ob = next_ob
        if done:
            num_trajectories += 1
            if budget_truncated:
                print('truncating final trajectory after reaching num_additional_steps')
            elif success:
                print('finished')
            new_replay_buffer['terminals'][new_replay_buffer.size - 1] = 1.0
            if 'masks' in new_replay_buffer:
                new_replay_buffer['masks'][new_replay_buffer.size - 1] = 0.0
            if 'valids' in new_replay_buffer:
                new_replay_buffer['valids'][new_replay_buffer.size - 1] = 0.0

            subgoals_buffers[task_key].append(subgoals_buffer)
            successes[task_key].append(success)
            task_success_rate = float(np.mean(successes[task_key]))
            rollout_step = config['num_train_steps'] + num_collected_steps

            fig = make_rollout_figure(
                all_cell_points,
                rollout_observations,
                subgoals_buffer,
                start,
                goal,
                (
                    f'DQC with proposer + filtering rollout, task {cur_task_id}, '
                    f'trajectory {num_trajectories}, success={success:.0f}'
                ),
            )
            rollout_image = wandb.Image(fig)
            plt.close(fig)
            del fig
            log_wandb(
                {
                    f'data_collection/task{cur_task_id}_success': success,
                    f'data_collection/task{cur_task_id}_success_rate': task_success_rate,
                    'data_collection/completed_trajectories': len(successes[task_key]),
                    'data_collection/additional_steps': num_collected_steps,
                    'data_collection/replay_buffer_size': new_replay_buffer.size,
                    f'data_collection/task{cur_task_id}_rollout': rollout_image,
                },
                step=rollout_step,
            )

            if budget_exhausted:
                break

            ob, _ = env.reset(options=dict(task_id=cur_task_id))
            rollout_observations = []
            subgoals_buffer = []
            subgoal = None
            to_subgoal = 0

final_success_metrics = {
    f'data_collection/final_task{cur_task_id}_success_rate': float(np.mean(successes[task_key])) if successes[task_key] else 0.0,
    'data_collection/final_completed_trajectories': len(successes[task_key]),
    'data_collection/final_additional_steps': num_collected_steps,
    'data_collection/final_replay_buffer_size': new_replay_buffer.size,
}
log_wandb(final_success_metrics, step=config['num_train_steps'] + num_collected_steps + 1)

fql_save_dir = pathlib.Path(args['save_dir']) / args['wandb_group'] / run_name
fql_save_dir.mkdir(parents=True, exist_ok=True)

replay_buffer_name = args['replay_buffer_name']
if replay_buffer_name is None:
    replay_buffer_name = f'data-task{cur_task_id}-{new_replay_buffer.size}.npz'
replay_buffer_path = _save_replay_buffer(new_replay_buffer, fql_save_dir, replay_buffer_name)
print(f'Saved replay buffer to {replay_buffer_path}')

##=========== TRAIN FQL ON COLLECTED REPLAY BUFFER ===========##

if new_replay_buffer.size == 0:
    raise RuntimeError('Replay buffer is empty — no data was collected. Check subgoal filtering or environment setup.')

fql_config = _build_fql_config(config, n_step=args['fql_n_step'], discount_override=args['fql_discount'], alpha_override=args['fql_alpha'])
idxs = np.random.randint(new_replay_buffer.size, size=1)
fql_example_batch = new_replay_buffer.sample(1, idxs=idxs)
fql_agent = agents['fql'].create(
    args['seed'],
    fql_example_batch['observations'],
    fql_example_batch['actions'],
    fql_config,
)

print(f'Training FQL for {args["fql_train_steps"]} steps from replay buffer')
print(f'Saving FQL checkpoints to {fql_save_dir}')
fql_step_offset = config['num_train_steps'] + num_collected_steps + 1
for fql_step in tqdm(range(1, args['fql_train_steps'] + 1)):
    if args['dataset_replace_interval'] > 0 and fql_step % args['dataset_replace_interval'] == 0 and len(datasets) > 1:
        dataset_idx = (dataset_idx + 1) % len(datasets)
        _, new_base_train, _ = make_env_and_datasets(
            config['env_name'],
            dataset_path=datasets[dataset_idx],
            # dataset_only=True,
            cur_env=env,
            use_oracle_reps=True,
        )
        current_base_data = _prepare_initial_replay_data(_dataset_mapping(new_base_train))

    batch = _proportional_sample(current_base_data, new_replay_buffer, total_base_size, fql_config['batch_size'],
                                 n_step=fql_config['n_step'], discount=fql_config['discount'])
    fql_agent, fql_info = fql_agent.update(batch)

    if fql_step == 1 or fql_step % args['fql_log_interval'] == 0:
        metrics = {f'fql/training/{key}': float(value) for key, value in fql_info.items()}
        metrics['fql/new_buffer_size'] = new_replay_buffer.size
        metrics['fql/total_virtual_size'] = total_base_size + new_replay_buffer.size
        log_wandb(metrics, step=fql_step_offset + fql_step)

    if args['fql_eval_interval'] > 0 and fql_step % args['fql_eval_interval'] == 0:
        eval_info, _, _ = evaluate(
            agent=fql_agent,
            env=env,
            num_eval_episodes=args['fql_eval_episodes'],
        )
        eval_metrics = {f'fql/eval/{k}': float(v) for k, v in eval_info.items()}
        log_wandb(eval_metrics, step=fql_step_offset + fql_step)

    if args['fql_save_interval'] > 0 and fql_step % args['fql_save_interval'] == 0:
        save_agent(fql_agent, str(fql_save_dir), fql_step)

wandb.finish()
