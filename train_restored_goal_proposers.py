from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from tqdm.auto import trange

from agents.goal_proposer import (
    GCFlowGoalProposerAgent,
    GoalProposerCGCDataset,
    latest_checkpoint_step,
    make_goal_proposer_config,
)
from utils.flax_utils import restore_agent, save_agent
from wrappers.datafuncs_utils import make_env_and_datasets


ENV_NAME = 'humanoidmaze-giant-navigate-v0'
ENV_DATASET_PATH = (
    '../../scratch/data/humanoidmaze-giant-navigate-v0/'
    'humanoidmaze-giant-navigate-100m-v0/humanoidmaze-giant-navigate-v0-003.npz'
)

ADDITIONAL_TRAIN_STEPS = 2_000_000
SAVE_INTERVAL = 250_000

BASE_CONFIG = dict(
    observations_key='oracle_reps',
    goal_key='actor_goals',
    hidden_dims=(256, 256, 256),
    layer_norm=True,
    lr=3e-4,
    flow_steps=10,
    seed=0,
    discount=0.995,
    backup_horizon=25,
    batch_size=256,
    log_interval=1000,
)

PROPOSER_SPECS = {
    'unconditioned_h100': dict(observation_conditioned=False, horizon_conditioned=False, seed_offset=0),
    'observation_h100': dict(observation_conditioned=True, horizon_conditioned=False, seed_offset=1),
    'observation_horizon_h1_100': dict(observation_conditioned=True, horizon_conditioned=True, seed_offset=2),
}


def restore_variant(name, config, dataset):
    example_batch = dataset.sample(config['batch_size'])
    agent = GCFlowGoalProposerAgent.create(example_batch, config, seed=config['seed'])
    checkpoint_step = latest_checkpoint_step(config['output_dir'])
    agent = restore_agent(agent, config['output_dir'], checkpoint_step)
    return agent, checkpoint_step


def train_variant(name, config, base_train_dataset, additional_steps, save_interval):
    dataset = GoalProposerCGCDataset(base_train_dataset, config=config)
    agent, start_step = restore_variant(name, config, dataset)
    target_step = start_step + additional_steps
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'{name}: continuing from checkpoint {start_step:,} to {target_step:,}')
    pbar = trange(1, additional_steps + 1, desc=name)
    last_info = None
    for local_step in pbar:
        absolute_step = start_step + local_step
        batch = dataset.sample(config['batch_size'])
        agent, info = agent.update(batch)
        last_info = info

        if local_step == 1 or local_step % config['log_interval'] == 0:
            pbar.set_description(
                f"{name} step={absolute_step:,} "
                f"loss={float(info['flow_loss']):.4f} "
                f"mae={float(info['velocity_mae']):.4f} "
                f"endpoint={float(info['endpoint_mse']):.4f}"
            )

        if save_interval and local_step % save_interval == 0:
            save_agent(agent, str(output_dir), absolute_step)

    if save_interval and additional_steps % save_interval != 0:
        save_agent(agent, str(output_dir), target_step)

    if last_info is not None:
        print(
            f"{name}: finished at {target_step:,} "
            f"loss={float(last_info['flow_loss']):.6f} "
            f"mae={float(last_info['velocity_mae']):.6f} "
            f"endpoint={float(last_info['endpoint_mse']):.6f}"
        )
    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--additional-steps', type=int, default=ADDITIONAL_TRAIN_STEPS)
    parser.add_argument('--save-interval', type=int, default=SAVE_INTERVAL)
    parser.add_argument('--seed', type=int, default=BASE_CONFIG['seed'])
    parser.add_argument(
        '--names',
        nargs='+',
        default=tuple(PROPOSER_SPECS),
        choices=tuple(PROPOSER_SPECS),
        help='Subset of proposer variants to continue.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    _, base_train_dataset, _ = make_env_and_datasets(
        ENV_NAME,
        dataset_path=ENV_DATASET_PATH,
        use_oracle_reps=True,
    )

    base_config = dict(BASE_CONFIG, seed=args.seed)
    for name in args.names:
        spec = PROPOSER_SPECS[name]
        config = make_goal_proposer_config(name, base_config=base_config, **spec)
        train_variant(name, config, base_train_dataset, args.additional_steps, args.save_interval)

    print('latest checkpoint steps:')
    for name in args.names:
        checkpoint_dir = Path('checkpoints/gc_flow_goal_proposer') / name
        print(f'{name}: {latest_checkpoint_step(checkpoint_dir):,}')
    print('jax devices:', jax.devices())


if __name__ == '__main__':
    main()
