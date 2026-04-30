from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'

DATASET_DIR = '../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/'

DATASET_PATH = DATASET_DIR + 'humanoidmaze-giant-navigate-v0-000.npz'

CKPT_NUM = 1_000_000

##=========== IMPORTS ===========##

import glob
import json
import os
import pathlib

import numpy as np
from tqdm import tqdm
from agents import agents
from agents.fql import get_config as get_fql_config
from utils.datasets import Dataset, GCDataset, HGCDataset, CGCDataset, ReplayBuffer
from wrappers.datafuncs_utils import make_env_and_datasets
from utils.evaluation import evaluate

import flax
import jax
import wandb

from utils.flax_utils import restore_agent, save_agent
import argparse


##=========== FLAGS ===========##

parser = argparse.ArgumentParser()
parser.add_argument('--replay_buffer_path', type=str, required=True, help='Path to the .npz replay buffer file.')
parser.add_argument('--restore_path', type=str, default=PATH, help='Path to the DQC checkpoint (used to read agent config).')
parser.add_argument('--ckpt_num', type=int, default=CKPT_NUM)
parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
parser.add_argument('--dataset_replace_interval', type=int, default=1000)
parser.add_argument('--num_datasets', type=int, default=None)
parser.add_argument('--env_name', type=str, default='humanoidmaze-giant-navigate-v0')
parser.add_argument('--task_id', type=int, default=1)
parser.add_argument('--fql_train_steps', type=int, default=1_000_000)
parser.add_argument('--fql_n_step', type=int, default=1)
parser.add_argument('--fql_discount', type=float, default=None, help='FQL discount; defaults to agent config discount if not set.')
parser.add_argument('--fql_alpha', type=float, default=None, help='FQL BC coefficient; defaults to agent config alpha if not set.')
parser.add_argument('--fql_best_of_n', type=int, default=None, help='FQL best-of-n actions; defaults to agent config best_of_n if not set.')
parser.add_argument('--fql_log_interval', type=int, default=1000)
parser.add_argument('--fql_eval_interval', type=int, default=50000)
parser.add_argument('--fql_eval_episodes', type=int, default=10)
parser.add_argument('--fql_save_interval', type=int, default=100000)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--wandb_project', type=str, default='aorl2')
parser.add_argument('--wandb_entity', type=str, default='moma1234')
parser.add_argument('--wandb_group', type=str, default='giant_fql_restored')
parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))
parser.add_argument('--save_dir', type=str, default='../../scratch/checkpoints/fql_restored')
parser.add_argument('--step_offset', type=int, default=0, help='Wandb step offset (e.g. set to num_train_steps + num_collected_steps).')

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

run_name = f"fql_restored_rb{pathlib.Path(args['replay_buffer_path']).stem}_seed{args['seed']}"
wandb.init(
    project=args['wandb_project'],
    entity=args['wandb_entity'],
    group=args['wandb_group'],
    name=run_name,
    mode=args['wandb_mode'],
    config=args,
    dir='../../scratch/wandb',
    settings=wandb.Settings(start_method='thread'),
)


def log_wandb(metrics, step=None):
    if wandb.run is not None:
        wandb.log(metrics, step=step)


##=========== LOAD AGENT CONFIG ===========##

flags_path = os.path.join(args['restore_path'], 'flags.json')
with open(flags_path, 'r') as f:
    saved_flags = json.load(f)
agent_config = saved_flags['agent']


##=========== HELPER FUNCTIONS ===========##

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


def _build_fql_config(config, n_step=1, discount_override=None, alpha_override=None, best_of_n_override=None):
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
    if best_of_n_override is not None:
        fql_config['best_of_n'] = int(best_of_n_override)
    return fql_config


def _n_step_from_dict(data, batch_size, n_step, discount):
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


##=========== LOAD REPLAY BUFFER ===========##

print(f'Loading replay buffer from {args["replay_buffer_path"]}')
rb_npz = np.load(args['replay_buffer_path'])
rb_dict = {k: np.asarray(v) for k, v in rb_npz.items()}
rb_size = len(rb_dict['observations'])
new_replay_buffer = ReplayBuffer.create_from_initial_dataset(rb_dict, size=rb_size)
print(f'Loaded replay buffer: {new_replay_buffer.size} transitions')

if new_replay_buffer.size == 0:
    raise RuntimeError('Replay buffer is empty.')

##=========== LOAD BASE DATASET ===========##

dataset_class_name = agent_config.get('dataset_class', 'GCDataset')
dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset, 'CGCDataset': CGCDataset}[dataset_class_name]

agent_config_for_ds = {**agent_config,
    'actor_p_curgoal': 0.5, 'actor_p_trajgoal': 0.5, 'actor_p_randomgoal': 0.0,
    'actor_geom_sample': 0.5, 'subgoal_steps': 100, 'train_goal_proposer': False,
}
dataset_npz = np.load(datasets[0])
base_ds = dataset_class(Dataset.create(**dict(dataset_npz)), config=agent_config_for_ds)
current_base_data = _prepare_initial_replay_data(_dataset_mapping(base_ds))
total_base_size = len(current_base_data['observations'])
print(f'Loaded base dataset: {total_base_size} transitions from {datasets[0]}')

##=========== LOAD ENV FOR EVAL ===========##

env, _, _ = make_env_and_datasets(args['env_name'], dataset_path=datasets[0], use_oracle_reps=True)

##=========== TRAIN FQL ===========##

fql_config = _build_fql_config(
    agent_config,
    n_step=args['fql_n_step'],
    discount_override=args['fql_discount'],
    alpha_override=args['fql_alpha'],
    best_of_n_override=args['fql_best_of_n'],
)

idxs = np.random.randint(new_replay_buffer.size, size=1)
fql_example_batch = new_replay_buffer.sample(1, idxs=idxs)
fql_agent = agents['fql'].create(
    args['seed'],
    fql_example_batch['observations'],
    fql_example_batch['actions'],
    fql_config,
)

save_dir = pathlib.Path(args['save_dir'])
save_dir.mkdir(parents=True, exist_ok=True)

print(f'Training FQL for {args["fql_train_steps"]} steps')
step_offset = args['step_offset']
for fql_step in tqdm(range(1, args['fql_train_steps'] + 1)):
    if args['dataset_replace_interval'] > 0 and fql_step % args['dataset_replace_interval'] == 0 and len(datasets) > 1:
        dataset_idx = (dataset_idx + 1) % len(datasets)
        dataset_npz = np.load(datasets[dataset_idx])
        base_ds = dataset_class(Dataset.create(**dict(dataset_npz)), config=agent_config_for_ds)
        current_base_data = _prepare_initial_replay_data(_dataset_mapping(base_ds))

    batch = _proportional_sample(
        current_base_data, new_replay_buffer, total_base_size, fql_config['batch_size'],
        n_step=fql_config['n_step'], discount=fql_config['discount'],
    )
    fql_agent, fql_info = fql_agent.update(batch)

    if fql_step == 1 or fql_step % args['fql_log_interval'] == 0:
        metrics = {f'fql/training/{k}': float(v) for k, v in fql_info.items()}
        metrics['fql/replay_buffer_size'] = new_replay_buffer.size
        metrics['fql/total_virtual_size'] = total_base_size + new_replay_buffer.size
        log_wandb(metrics, step=step_offset + fql_step)

    if args['fql_eval_interval'] > 0 and fql_step % args['fql_eval_interval'] == 0:
        eval_info, _, _ = evaluate(
            agent=fql_agent,
            env=env,
            num_eval_episodes=args['fql_eval_episodes'],
        )
        log_wandb({f'fql/eval/{k}': float(v) for k, v in eval_info.items()}, step=step_offset + fql_step)

    if args['fql_save_interval'] > 0 and fql_step % args['fql_save_interval'] == 0:
        save_agent(fql_agent, str(save_dir), fql_step)

wandb.finish()
