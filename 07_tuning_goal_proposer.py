from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/aorl2/2026-04-08-00/2026-04-08-00.b7bf8a914965d2ce2cdfd7704faa38b5fee704b874bc391d21e3b9137701759c/'

CKPT_NUM = 1000000

##=========== IMPORTS ===========##

import json
import os
import pathlib
import time

import numpy as np
from tqdm import tqdm
from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, CGCDataset
from wrappers.datafuncs_utils import make_env_and_datasets
from utils.networks import ActorVectorField

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb

from utils.datasets import GCDataset
from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import ActorVectorField, MLP
from wrappers.datafuncs_utils import make_env_and_datasets

from utils.flax_utils import restore_agent
import argparse


##=========== FLAGS ===========##

parser = argparse.ArgumentParser()
parser.add_argument('--subgoal_steps', type=int, default=100)
parser.add_argument('--steps_to_subgoal', type=int, default=25)
parser.add_argument('--num_train_steps', type=int, default=100000)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--num_trial_steps', type=int, default=2000)
parser.add_argument('--num_subgoals', type=int, default=128)
parser.add_argument('--mult_factor', type=float, default=0.9)
parser.add_argument('--additive_factor', type=float, default=0.0)
parser.add_argument('--A_B_factor', type=float, default=1.0)
parser.add_argument('--B_C_factor', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--wandb_project', type=str, default='aorl2')
parser.add_argument('--wandb_entity', type=str, default='moma1234')
parser.add_argument('--wandb_group', type=str, default='tuning_goal_proposer')
parser.add_argument('--wandb_mode', type=str, default=os.environ.get('WANDB_MODE', 'online'))

args = vars(parser.parse_args())

run_name = (
    f"goal_proposer_sg{args['subgoal_steps']}_train{args['num_train_steps']}_"
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
agent_config['subgoal_steps'] = args['subgoal_steps']

# %%
dataset_class_name = agent_config.get('dataset_class', 'GCDataset')
dataset_class = {
    'GCDataset': GCDataset,
    'HGCDataset': HGCDataset,
    'CGCDataset': CGCDataset,
}[dataset_class_name]

dataset_path = os.path.join(PATH, 'data-100000.npz')
dataset_npz = np.load(dataset_path)
train_dataset = dataset_class(Dataset.create(**dict(dataset_npz)), config=agent_config)

# seed = saved_flags.get('seed', 0)
seed = args['seed']
example_batch = train_dataset.sample(1)

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

config = dict(
    env_name='humanoidmaze-large-navigate-v0',
    # dataset_path='../../scratch/aorl2/YOUR_RUN_DIR/data-1000000.npz',
    dataset_path='../../scratch/data/humanoidmaze-large-navigate-v0/humanoidmaze-large-navigate-v0seed-0.npz',
    observations_key='oracle_reps', # 'observations',
    goal_key='actor_goals',
    actions_key='low_actor_goals', #'actions',
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
    discount=0.995,
    flow_steps=10,
    backup_horizon=25,
    goal_conditioned=False,
)

env, base_train_dataset, val_dataset = make_env_and_datasets(
    config['env_name'],
    dataset_path=config['dataset_path'],
    use_oracle_reps=True,
)
train_dataset = CGCDataset(base_train_dataset, config=config)

class GCFlowGoalProposerAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def flow_loss(self, batch, grad_params=None, rng=None):
        observations = batch[self.config['observations_key']]
        goals = batch[self.config['goal_key']] if self.config['goal_conditioned'] else None
        target_actions = batch[self.config['actions_key']]

        batch_size, action_dim = target_actions.shape
        rng = self.rng if rng is None else rng
        x_rng, t_rng = jax.random.split(rng)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * target_actions
        vel = target_actions - x_0

        pred_vel = self.network(
            observations,
            goals=goals,
            actions=x_t,
            times=t,
            params=grad_params,
        )
        loss = jnp.mean(jnp.square(pred_vel - vel))
        mae = jnp.mean(jnp.abs(pred_vel - vel))
        return loss, {
            'flow_loss': loss,
            'velocity_mae': mae,
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.flow_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        info['step'] = new_network.step
        return self.replace(rng=new_rng, network=new_network), info

    @jax.jit
    def sample_actions(self, observations, goals, rng):
        single_example = observations.ndim == 1
        if not self.config['goal_conditioned']:
            goals = None
        if single_example:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        x = jax.random.normal(rng, (observations.shape[0], self.config['action_dim']))

        for i in range(self.config['flow_steps']):
            t = jnp.full((observations.shape[0], 1), i / self.config['flow_steps'])
            vels = self.network(observations, goals=goals, actions=x, times=t)
            x = x + vels / self.config['flow_steps']

        return x[0] if single_example else x

    @classmethod
    def create(cls, example_batch, config):
        config = dict(config)
        config.setdefault('goal_conditioned', True)
        rng = jax.random.PRNGKey(args['seed'])
        rng, init_rng = jax.random.split(rng)
        action_dim = example_batch[config['actions_key']].shape[-1]
        model = ActorVectorField(
            hidden_dims=tuple(config['hidden_dims']),
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )
        init_goals = example_batch[config['goal_key']] if config['goal_conditioned'] else None
        params = model.init(
            init_rng,
            example_batch[config['observations_key']],
            goals=init_goals,
            actions=example_batch[config['actions_key']],
            times=example_batch[config['actions_key']][..., :1],
        )['params']
        network = TrainState.create(model, params, tx=optax.adam(config['lr']))
        config['action_dim'] = action_dim
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))

##=========== TRAIN GOAL PROPOSER ===========##

example_batch = train_dataset.sample(1)
flow_agent = GCFlowGoalProposerAgent.create(example_batch, config)
jax.tree_util.tree_map(lambda x: x.shape, flow_agent.network.params)

flow_loss_history = []
velocity_mae_history = []

for step in range(1, config['num_train_steps'] + 1):
    batch = train_dataset.sample(config['batch_size'])
    flow_agent, info = flow_agent.update(batch)

    flow_loss_history.append(float(info['flow_loss']))
    velocity_mae_history.append(float(info['velocity_mae']))

    if step == 1 or step % config['log_interval'] == 0:
        log_wandb(
            {
                'goal_proposer/flow_loss': float(info['flow_loss']),
                'goal_proposer/velocity_mae': float(info['velocity_mae']),
            },
            step=step,
        )
        print(
            f"step={step:05d} flow_loss={flow_loss_history[-1]:.6f} velocity_mae={velocity_mae_history[-1]:.6f}"
        )

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

def sample_n(ob, goal, n, sample_rng, agent=flow_agent):
    ob_xy = ob[:2]
    obs = np.repeat(ob_xy[None], n, axis=0)
    goals = np.repeat(goal[None], n, axis=0)

    return flow_agent.sample_actions(obs, goals, sample_rng)

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

        ##=========== WITH SUBGOALS ===========##

        replay_buffer = []
        subgoals_buffer = []
        success = 0.0

        ob, _ = env.reset(options=dict(task_id=cur_task_id))
        subgoal = None
        to_subgoal = 0

        for s in tqdm(range(args['num_trial_steps'])):
            replay_buffer.append(ob)

            if subgoal is None or to_subgoal == args['steps_to_subgoal']:
                rng, sample_rng = jax.random.split(rng)
                subgoals = np.asarray(sample_n(ob, goal, args['num_subgoals'], sample_rng))

                gamma_to_subgoal, gamma_to_goal, ob_to_goal = dynamical_distance(ob, subgoals, goal)
                gamma_to_subgoal = np.asarray(gamma_to_subgoal).reshape(-1)
                gamma_to_goal = np.asarray(gamma_to_goal).reshape(-1)
                ob_to_goal = float(np.asarray(ob_to_goal).reshape(-1)[0])

                mask = gamma_to_goal < ob_to_goal * args['mult_factor'] + args['additive_factor']
                if not np.any(mask):
                    print(f'no improving subgoal found at step {s}')
                    break

                filtered_subgoals = subgoals[mask]
                filtered_scores = (args['A_B_factor'] * gamma_to_subgoal + args['B_C_factor'] * gamma_to_goal)[mask]
                subgoal = filtered_subgoals[int(np.argmin(filtered_scores))]
                subgoals_buffer.append(subgoal)
                to_subgoal = 0

            action_rng, rng = jax.random.split(rng)
            action = dqc_agent.sample_actions(observations=ob, goals=subgoal, seed=action_rng)
            to_subgoal += 1
            ob, reward, terminated, truncated, _ = env.step(action)

            if np.linalg.norm(ob[:2] - subgoal) < 0.1:
                subgoal = None
            
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
        rollout_step = config['num_train_steps'] + sum(len(v) for v in successes.values())

        fig = make_rollout_figure(
            all_cell_points,
            replay_buffer,
            subgoals_buffer,
            start,
            goal,
            (
                f'DQC with proposer + filtering rollout, task {cur_task_id}, trial {trial + 1}, '
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
log_wandb(final_success_metrics, step=config['num_train_steps'] + sum(len(v) for v in successes.values()) + 1)
wandb.finish()
