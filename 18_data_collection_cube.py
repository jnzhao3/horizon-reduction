from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/dqc-reproduce/sd100001s_33728300.0.33728299.1.20260425_014426/'

DATASET_PATH = '../../scratch/data/cube-quadruple-play-v0/cube-quadruple-play-v0.npz'

CKPT_NUM = 1_000_000

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
parser.add_argument('--subgoal_steps', type=int, default=250)
parser.add_argument('--steps_to_subgoal', type=int, default=25)
parser.add_argument('--num_train_steps', type=int, default=3000000)
parser.add_argument('--num_additional_steps', type=int, default=1000000)
parser.add_argument('--fql_train_steps', type=int, default=1000000)
parser.add_argument('--fql_chunk_size', type=int, default=5)
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
parser.add_argument('--wandb_group', type=str, default='cube_data_collection')
parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))
parser.add_argument('--save_dir', type=str, default='../../scratch/checkpoints/data_collection_cube')
parser.add_argument('--replay_buffer_name', type=str, default=None)
parser.add_argument('--restore_path', type=str, default=PATH)
parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--dataset_replace_interval', type=int, default=1000)
parser.add_argument('--num_datasets', type=int, default=None)
parser.add_argument('--ckpt_num', type=int, default=CKPT_NUM)
parser.add_argument('--env_name', type=str, default='cube-quadruple-play-oraclerep-v0')
parser.add_argument('--flow_restore_path', type=str, default='../../scratch/checkpoints/cube_quadruple_horizon_subgoal_proposer')
parser.add_argument('--flow_ckpt_num', type=int, default=1_050_000)

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
    f"cube_goal_proposer_sg{args['subgoal_steps']}_train{args['num_train_steps']}_"
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


def make_rollout_figure(rollout_oracles, subgoals_buffer, start_oracle, goal_oracle, num_cubes, title=''):
    traj = np.array(rollout_oracles, dtype=np.float32).reshape(-1, num_cubes, 3) if rollout_oracles else np.zeros((0, num_cubes, 3))
    sgs = np.array(subgoals_buffer, dtype=np.float32).reshape(-1, num_cubes, 3) if subgoals_buffer else np.zeros((0, num_cubes, 3))
    start_xyz = np.array(start_oracle, dtype=np.float32).reshape(num_cubes, 3)
    goal_xyz = np.array(goal_oracle, dtype=np.float32).reshape(num_cubes, 3)

    fig, axes = plt.subplots(1, num_cubes, figsize=(4 * num_cubes, 4), squeeze=False)
    axes = axes[0]
    for i in range(num_cubes):
        ax = axes[i]
        if len(traj):
            ax.scatter(traj[:, i, 0], traj[:, i, 1], c=np.arange(len(traj)), cmap='viridis', s=5, zorder=2)
        if len(sgs):
            ax.scatter(sgs[:, i, 0], sgs[:, i, 1], c='orange', s=25, zorder=4)
        ax.scatter(*start_xyz[i, :2], marker='x', c='red', s=80, zorder=5, label='start')
        ax.scatter(*goal_xyz[i, :2], marker='*', c='green', s=120, zorder=5, label='goal')
        ax.set_title(f'cube {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
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
agent_config['subgoal_steps'] = args['subgoal_steps']
agent_config['train_goal_proposer'] = False

dataset_npz = np.load(datasets[0])
train_dataset = dataset_class(Dataset.create(**dict(dataset_npz)), config=agent_config)

seed = args['seed']
example_batch = train_dataset.sample(1)

first_agent = agents[agent_config['agent_name']].create(seed, example_batch, agent_config)
first_agent = restore_agent(first_agent, args['restore_path'], args['ckpt_num'])

print(f'Restored first_agent from checkpoint {args["ckpt_num"]}')

dqc_agent = first_agent

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
    observation_conditioned=True,
    horizon_conditioned=True,
    horizon_scale=float(args['subgoal_steps']),
    min_horizon_steps=1,
    max_horizon_steps=args['subgoal_steps'],
)

env, base_train_dataset, val_dataset = make_env_and_datasets(
    config['env_name'],
    dataset_path=datasets[0],
    use_oracle_reps=True,
)
num_cubes = int(env.unwrapped._num_cubes)
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


def _build_fql_config(config, chunk_size=5, n_step=1, discount_override=None, alpha_override=None):
    fql_config = get_fql_config()
    fql_config['agent_name'] = 'fql'
    fql_config['batch_size'] = int(config['batch_size'])
    fql_config['discount'] = float(discount_override if discount_override is not None else config['discount'])
    fql_config['flow_steps'] = int(config['flow_steps'])
    fql_config['horizon_length'] = int(chunk_size)
    fql_config['action_chunking'] = True
    fql_config['encoder'] = None
    fql_config['n_step'] = int(n_step)
    if alpha_override is not None:
        fql_config['alpha'] = float(alpha_override)
    return fql_config


def _sample_sequence_from_dict(data, batch_size, sequence_length, discount):
    """Sample action-chunked sequences from a plain dict of arrays."""
    size = len(data['observations'])
    idxs = np.random.randint(size - sequence_length + 1, size=batch_size)

    obs_dim = data['observations'].shape[-1]
    act_dim = data['actions'].shape[-1]

    rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
    masks = np.ones((batch_size, sequence_length), dtype=np.float32)
    terminals = np.zeros((batch_size, sequence_length), dtype=np.float32)
    actions = np.zeros((batch_size, sequence_length, act_dim), dtype=data['actions'].dtype)
    next_obs = np.zeros((batch_size, sequence_length, obs_dim), dtype=data['observations'].dtype)

    for i in range(sequence_length):
        cur_idxs = idxs + i
        actions[:, i, :] = data['actions'][cur_idxs]
        if i == 0:
            rewards[:, 0] = data['rewards'][cur_idxs]
            masks[:, 0] = data['masks'][cur_idxs]
            terminals[:, 0] = data['terminals'][cur_idxs]
            next_obs[:, 0, :] = data['next_observations'][cur_idxs]
        else:
            valid_i = 1.0 - terminals[:, i - 1]
            rewards[:, i] = rewards[:, i - 1] + data['rewards'][cur_idxs] * (discount ** i) * valid_i
            masks[:, i] = np.minimum(masks[:, i - 1], data['masks'][cur_idxs]) * valid_i + masks[:, i - 1] * (1.0 - valid_i)
            terminals[:, i] = np.maximum(terminals[:, i - 1], data['terminals'][cur_idxs])
            next_obs[:, i, :] = (
                data['next_observations'][cur_idxs] * valid_i[:, None]
                + next_obs[:, i - 1, :] * (1.0 - valid_i[:, None])
            )

    result = {k: v[idxs].copy() for k, v in data.items()}
    result['observations'] = data['observations'][idxs].copy()
    result['actions'] = actions
    result['rewards'] = rewards
    result['masks'] = masks
    result['next_observations'] = next_obs
    return result


def _proportional_sample(base_data, new_buffer, total_base_size, batch_size,
                         action_chunking=False, horizon_length=1, discount=0.99):
    R = new_buffer.size
    total = total_base_size + R
    n_new = int(np.random.binomial(batch_size, R / total)) if R > 0 else 0
    n_base = batch_size - n_new

    parts = []
    if n_base > 0:
        if action_chunking:
            parts.append(_sample_sequence_from_dict(base_data, n_base, horizon_length, discount))
        else:
            idxs = np.random.randint(len(base_data['observations']), size=n_base)
            parts.append({k: v[idxs] for k, v in base_data.items()})
    if n_new > 0:
        if action_chunking:
            parts.append(new_buffer.sample_sequence(n_new, horizon_length, discount))
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

proposer_example_batch = proposer_train_dataset.sample(1)
flow_agent = GCFlowGoalProposerAgent.create(proposer_example_batch, config)
flow_agent = restore_agent(flow_agent, args['flow_restore_path'], args['flow_ckpt_num'])
print(f'Restored flow_agent from {args["flow_restore_path"]} checkpoint {args["flow_ckpt_num"]}')

##=========== UTILITIES ===========##

def sigmoid(x):
    x = np.asarray(x)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )

_XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
_XYZ_SCALER = 10.0
_QPOS_OBJ_START = 14
_QPOS_CUBE_LEN = 7

def obs_to_oracle_rep(ob):
    return np.asarray(to_oracle_reps(np.asarray(ob)[None], env=env))[0]

def make_obs_at_oracle_rep(ob, oracle_rep):
    """Return a copy of ob with cube positions replaced by those encoded in oracle_rep."""
    ob_virtual = ob.copy()
    cube_xyzs = oracle_rep.reshape(num_cubes, 3) / _XYZ_SCALER + _XYZ_CENTER
    for i in range(num_cubes):
        start = _QPOS_OBJ_START + i * _QPOS_CUBE_LEN
        ob_virtual[start:start + 3] = cube_xyzs[i]
    return ob_virtual

def dynamical_distance(ob, subgoals, goal_oracle_rep):
    discount = dqc_agent.config['discount']

    # gamma_to_subgoal: V(ob → subgoal)
    all_obs = np.repeat(ob[None], len(subgoals), axis=0)
    ob_to_subgoal_vs = sigmoid(np.asarray(dqc_agent.network.select('value')(all_obs, subgoals)))
    gamma_to_subgoal = np.log(np.clip(ob_to_subgoal_vs, 1e-6, 1.0)) / np.log(discount)

    # gamma_to_goal: V(subgoal → goal)
    subgoal_obs = np.stack([make_obs_at_oracle_rep(ob, sg) for sg in subgoals])
    goal_rep_batch = np.repeat(goal_oracle_rep[None], len(subgoals), axis=0)
    subgoal_to_goal_vs = sigmoid(np.asarray(dqc_agent.network.select('value')(subgoal_obs, goal_rep_batch)))
    gamma_to_goal = np.log(np.clip(subgoal_to_goal_vs, 1e-6, 1.0)) / np.log(discount)

    # ob_to_goal: V(ob → goal)
    ob_to_goal_v = sigmoid(np.asarray(dqc_agent.network.select('value')(ob[None], goal_oracle_rep[None]))).reshape(-1)[0]
    ob_to_goal = float(np.log(np.clip(ob_to_goal_v, 1e-6, 1.0)) / np.log(discount))

    return gamma_to_subgoal.reshape(-1), gamma_to_goal.reshape(-1), ob_to_goal

def sample_n(ob, n, sample_rng):
    oracle_rep = obs_to_oracle_rep(ob)
    obs = np.repeat(oracle_rep[None], n, axis=0)
    return flow_agent.sample_actions(
        observations=obs,
        horizons=float(args['subgoal_steps']),
        rng=sample_rng,
    )

##=========== ROLLOUTS ===========##

subgoals_buffers = {}
successes = {}
rng = jax.random.PRNGKey(args['seed'])
task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos

if args['task_id'] < 1 or args['task_id'] > len(task_infos):
    raise ValueError(f'--task_id must be in [1, {len(task_infos)}], got {args["task_id"]}')

cur_task_id = args['task_id']
task_info = task_infos[cur_task_id - 1]
task_key = f'task_{cur_task_id}'

ob, info = env.reset(options=dict(task_info=task_info))
goal_oracle = np.asarray(info['goal'], dtype=np.float32)
start_oracle = obs_to_oracle_rep(ob)

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

print(f'running task {cur_task_id} ({task_info["task_name"]})')
print(f'dataset: {single_dataset_size} transitions; new buffer capacity={args["num_additional_steps"]}')

num_collected_steps = 0
num_trajectories = 0
num_subgoal_selections = 0
num_random_subgoal_selections = 0
collection_stats = get_statistics_class(config['env_name'])(env=env)

with tqdm(total=args['num_additional_steps']) as pbar:
    ob, info = env.reset(options=dict(task_info=task_info))
    goal_oracle = np.asarray(info['goal'], dtype=np.float32)
    rollout_oracles = []
    subgoals_buffer = []
    subgoal = None
    to_subgoal = 0
    last_mask_size = 0

    while num_collected_steps < args['num_additional_steps']:
        oracle_rep = obs_to_oracle_rep(ob)
        rollout_oracles.append(oracle_rep.copy())

        if subgoal is None or to_subgoal == args['steps_to_subgoal']:
            rng, sample_rng = jax.random.split(rng)
            subgoals = np.asarray(sample_n(ob, args['num_subgoals'], sample_rng))

            gamma_to_subgoal, gamma_to_goal, ob_to_goal = dynamical_distance(ob, subgoals, goal_oracle)
            gamma_to_subgoal = np.asarray(gamma_to_subgoal).reshape(-1)
            gamma_to_goal = np.asarray(gamma_to_goal).reshape(-1)

            mask = gamma_to_goal < ob_to_goal * args['mult_factor'] + args['additive_factor']
            last_mask_size = int(np.sum(mask))
            num_subgoal_selections += 1
            if not np.any(mask):
                rng, key = jax.random.split(rng)
                subgoal = subgoals[jax.random.randint(key, (), 0, len(subgoals))]
                num_random_subgoal_selections += 1
            else:
                filtered_subgoals = subgoals[mask]
                filtered_scores = (args['A_B_factor'] * gamma_to_subgoal + args['B_C_factor'] * gamma_to_goal)[mask]
                subgoal = filtered_subgoals[int(np.argmin(filtered_scores))]

            subgoals_buffer.append(subgoal)
            to_subgoal = 0

        # cube DQC uses policy_chunk_size=5; iterate over the action chunk
        action_rng, rng = jax.random.split(rng)
        action_chunk = dqc_agent.sample_actions(observations=ob, goals=subgoal, seed=action_rng)
        action_chunk = np.asarray(action_chunk)
        action_dim = env.action_space.shape[0]
        if action_chunk.ndim == 1 and action_chunk.shape[0] > action_dim:
            action_chunk = action_chunk.reshape(-1, action_dim)
        action_seq = [action_chunk] if action_chunk.ndim == 1 else list(action_chunk)

        done = False
        budget_exhausted = False
        for single_action in action_seq:
            if num_collected_steps >= args['num_additional_steps']:
                budget_exhausted = True
                break

            single_action = np.clip(single_action, -1, 1)
            to_subgoal += 1
            next_ob, reward, terminated, truncated, info = env.step(single_action)
            num_collected_steps += 1
            budget_exhausted = num_collected_steps >= args['num_additional_steps']
            next_oracle_rep = obs_to_oracle_rep(next_ob)
            success = float(terminated or np.linalg.norm(next_oracle_rep - goal_oracle) < 0.04)
            budget_truncated = budget_exhausted and not (terminated or truncated or bool(success))
            done = terminated or truncated or budget_truncated or bool(success)

            transition = _make_transition_for_buffer(
                new_replay_buffer,
                ob=ob,
                action=single_action,
                reward=reward,
                done=done,
                truncated=truncated or budget_truncated,
                next_ob=next_ob,
                info=info,
                env=env,
            )
            new_replay_buffer.add_transition(transition)
            collection_stats.log_episode(ob, single_action)
            pbar.update(1)
            ob = next_ob

            if done:
                break

        if args['data_log_interval'] > 0 and num_collected_steps % args['data_log_interval'] == 0:
            log_wandb(
                {
                    **{f'data_collection/{k}': v for k, v in collection_stats.get_statistics().items()},
                    'data_collection/random_subgoal_frac': num_random_subgoal_selections / num_subgoal_selections,
                    'data_collection/mask_size': last_mask_size,
                },
                step=num_collected_steps,
            )

        if np.linalg.norm(obs_to_oracle_rep(ob) - subgoal) < 0.3:
            subgoal = None

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


            fig = make_rollout_figure(
                rollout_oracles,
                subgoals_buffer,
                start_oracle,
                goal_oracle,
                num_cubes,
                title=(
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
                step=num_collected_steps,
            )

            if budget_exhausted:
                break

            ob, info = env.reset(options=dict(task_info=task_info))
            goal_oracle = np.asarray(info['goal'], dtype=np.float32)
            start_oracle = obs_to_oracle_rep(ob)
            rollout_oracles = []
            subgoals_buffer = []
            subgoal = None
            to_subgoal = 0

final_success_metrics = {
    f'data_collection/final_task{cur_task_id}_success_rate': float(np.mean(successes[task_key])) if successes[task_key] else 0.0,
    'data_collection/final_completed_trajectories': len(successes[task_key]),
    'data_collection/final_additional_steps': num_collected_steps,
    'data_collection/final_replay_buffer_size': new_replay_buffer.size,
}
log_wandb(final_success_metrics, step=num_collected_steps + 1)

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

fql_config = _build_fql_config(config, chunk_size=args['fql_chunk_size'], n_step=args['fql_n_step'],
                               discount_override=args['fql_discount'], alpha_override=args['fql_alpha'])
if fql_config['action_chunking']:
    fql_example_batch = new_replay_buffer.sample_sequence(1, fql_config['horizon_length'], fql_config['discount'])
else:
    idxs = np.random.randint(new_replay_buffer.size, size=1)
    fql_example_batch = new_replay_buffer.sample(1, idxs=idxs)
fql_agent = agents['fql'].create(
    args['seed'],
    fql_example_batch['observations'],
    fql_example_batch['actions'],
    fql_config,
)

print(f'Training FQL for {args["fql_train_steps"]} steps from replay buffer')
fql_step_offset = num_collected_steps + 1
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
                                 action_chunking=fql_config['action_chunking'],
                                 horizon_length=fql_config['horizon_length'],
                                 discount=fql_config['discount'])
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

wandb.finish()
