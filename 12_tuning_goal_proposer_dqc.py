from __future__ import annotations

##=========== CONSTANTS ===========##

PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'

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
parser.add_argument('--min_horizon_steps', type=int, default=1)
parser.add_argument('--horizon_conditioned', action='store_true')
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
    min_horizon_steps=args['min_horizon_steps'],
    max_horizon_steps=args['subgoal_steps'],
    discount=0.995,
    flow_steps=10,
    backup_horizon=25,
    goal_conditioned=False,
    observation_conditioned=False,
    horizon_conditioned=args['horizon_conditioned'],
    horizon_key='horizons',
    horizon_scale=args['subgoal_steps'],
)

class GoalProposerCGCDataset(CGCDataset):
    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if not self.config.get('horizon_conditioned', False):
            return super().sample(batch_size, idxs=idxs, evaluation=evaluation)

        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        min_horizon = int(self.config.get('min_horizon_steps', 1))
        max_horizon = int(self.config.get('max_horizon_steps', self.config['subgoal_steps']))
        horizons = np.random.randint(min_horizon, max_horizon + 1, size=batch_size)
        target_idxs = np.minimum(idxs + horizons, final_state_idxs)

        if 'oracle_reps' in self.dataset:
            batch[self.config['actions_key']] = self.dataset['oracle_reps'][target_idxs]
        else:
            batch[self.config['actions_key']] = self.get_observations(target_idxs)

        if self.config.get('goal_conditioned', False):
            actor_goal_idxs = self.sample_goals(
                idxs,
                self.config['actor_p_curgoal'],
                self.config['actor_p_trajgoal'],
                self.config['actor_p_randomgoal'],
                self.config['actor_geom_sample'],
            )
            if 'oracle_reps' in self.dataset:
                batch[self.config['goal_key']] = self.dataset['oracle_reps'][actor_goal_idxs]
            else:
                batch[self.config['goal_key']] = self.get_observations(actor_goal_idxs)

        batch[self.config['horizon_key']] = horizons[:, None].astype(np.float32)
        return batch


env, base_train_dataset, val_dataset = make_env_and_datasets(
    config['env_name'],
    dataset_path=config['dataset_path'],
    use_oracle_reps=True,
)
train_dataset = GoalProposerCGCDataset(base_train_dataset, config=config)

class GCFlowGoalProposerAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _get_horizons(self, batch, batch_size):
        if not self.config['horizon_conditioned']:
            return None

        horizon_key = self.config.get('horizon_key', 'horizons')
        if horizon_key in batch:
            horizons = jnp.asarray(batch[horizon_key], dtype=jnp.float32)
        else:
            horizons = jnp.full((batch_size, 1), float(self.config['subgoal_steps']), dtype=jnp.float32)
        if horizons.ndim == 1:
            horizons = horizons[:, None]
        return horizons

    def _encode_horizons(self, horizons):
        horizon_scale = float(self.config.get('horizon_scale', self.config.get('subgoal_steps', 1)))
        horizons = jnp.asarray(horizons, dtype=jnp.float32)
        if horizons.ndim == 1:
            horizons = horizons[:, None]
        linear = horizons / horizon_scale
        log_linear = jnp.log1p(horizons) / jnp.log1p(horizon_scale)
        return jnp.concatenate([linear, log_linear], axis=-1)

    def _append_horizons(self, observations, horizons):
        if not self.config['horizon_conditioned']:
            return observations
        return jnp.concatenate([observations, self._encode_horizons(horizons)], axis=-1)

    def flow_loss(self, batch, grad_params=None, rng=None):
        target_actions = jnp.asarray(batch[self.config['actions_key']], dtype=jnp.float32)
        batch_size, action_dim = target_actions.shape

        if self.config['observation_conditioned']:
            observations = jnp.asarray(batch[self.config['observations_key']], dtype=jnp.float32)
        else:
            observations = jnp.zeros((batch_size, 0), dtype=jnp.float32)

        horizons = self._get_horizons(batch, batch_size)
        observations = self._append_horizons(observations, horizons)

        goals = batch[self.config['goal_key']] if self.config['goal_conditioned'] else None
        goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)

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
        endpoint_mse = jnp.mean(jnp.square((x_t + (1.0 - t) * pred_vel) - target_actions))
        return loss, {
            'flow_loss': loss,
            'velocity_mae': mae,
            'endpoint_mse': endpoint_mse,
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
    def _sample_actions(self, observations, goals, rng):
        x = jax.random.normal(rng, (observations.shape[0], self.config['action_dim']))

        for i in range(self.config['flow_steps']):
            t = jnp.full((observations.shape[0], 1), i / self.config['flow_steps'])
            vels = self.network(observations, goals=goals, actions=x, times=t)
            x = x + vels / self.config['flow_steps']

        return x

    def sample_actions(self, observations=None, goals=None, rng=None, num_samples=None, horizons=None):
        if rng is None:
            rng = self.rng

        if self.config['observation_conditioned']:
            if observations is None:
                raise ValueError('observations are required when observation_conditioned=True')
            observations = np.asarray(observations, dtype=np.float32)
            single_example = observations.ndim == 1
            if single_example:
                observations = observations[None, ...]
            return_single = single_example and num_samples is None
        else:
            if num_samples is not None:
                batch_size = int(num_samples)
            elif goals is not None and np.asarray(goals).ndim > 1:
                batch_size = int(np.asarray(goals).shape[0])
            elif observations is not None and np.asarray(observations).ndim > 1:
                batch_size = int(np.asarray(observations).shape[0])
            elif horizons is not None and np.asarray(horizons).ndim > 0:
                batch_size = int(np.asarray(horizons).shape[0])
            else:
                batch_size = 1
            observations = np.zeros((batch_size, 0), dtype=np.float32)
            single_example = batch_size == 1
            return_single = single_example and num_samples is None

        if self.config['horizon_conditioned']:
            if horizons is None:
                horizons = np.full((observations.shape[0], 1), float(self.config['subgoal_steps']), dtype=np.float32)
            else:
                horizons = np.asarray(horizons, dtype=np.float32)
                if horizons.ndim == 0:
                    horizons = np.full((observations.shape[0], 1), float(horizons), dtype=np.float32)
                elif horizons.ndim == 1:
                    horizons = horizons[:, None]
                if horizons.shape[0] == 1 and observations.shape[0] != 1:
                    horizons = np.repeat(horizons, observations.shape[0], axis=0)
                elif horizons.shape[0] != observations.shape[0]:
                    raise ValueError(
                        f"horizons batch size must match observations batch size: "
                        f"{horizons.shape[0]} vs {observations.shape[0]}"
                    )
            observations = np.asarray(self._append_horizons(observations, horizons), dtype=np.float32)

        if not self.config['goal_conditioned']:
            goals = None
        elif goals is None:
            raise ValueError('goals are required when goal_conditioned=True')
        elif goals is not None:
            goals = np.asarray(goals, dtype=np.float32)
            if goals.ndim == 1:
                goals = goals[None, ...]

            if single_example and goals.shape[0] != observations.shape[0]:
                observations = np.repeat(observations, goals.shape[0], axis=0)
                return_single = False
            elif goals.shape[0] != observations.shape[0]:
                raise ValueError(
                    f"goals batch size must match observations batch size: {goals.shape[0]} vs {observations.shape[0]}"
                )

        x = self._sample_actions(observations, goals, rng)
        return x[0] if return_single else x

    @classmethod
    def create(cls, example_batch, config, seed=None):
        config = dict(config)
        config.setdefault('goal_conditioned', True)
        config.setdefault('observation_conditioned', True)
        config.setdefault('horizon_conditioned', False)
        config.setdefault('horizon_key', 'horizons')
        config.setdefault('horizon_scale', config.get('subgoal_steps', 1))
        seed = int(config.get('seed', 0) if seed is None else seed)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = example_batch[config['actions_key']].shape[-1]
        model = ActorVectorField(
            hidden_dims=tuple(config['hidden_dims']),
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )
        init_goals = example_batch[config['goal_key']] if config['goal_conditioned'] else None
        init_observations = (
            example_batch[config['observations_key']]
            if config['observation_conditioned']
            else jnp.zeros((*example_batch[config['actions_key']].shape[:-1], 0), dtype=jnp.float32)
        )
        if config['horizon_conditioned']:
            batch_size = example_batch[config['actions_key']].shape[0]
            if config['horizon_key'] in example_batch:
                init_horizons = jnp.asarray(example_batch[config['horizon_key']], dtype=jnp.float32)
            else:
                init_horizons = jnp.full((batch_size, 1), float(config['subgoal_steps']), dtype=jnp.float32)
            if init_horizons.ndim == 1:
                init_horizons = init_horizons[:, None]
            horizon_scale = float(config['horizon_scale'])
            horizon_features = jnp.concatenate(
                [
                    init_horizons / horizon_scale,
                    jnp.log1p(init_horizons) / jnp.log1p(horizon_scale),
                ],
                axis=-1,
            )
            init_observations = jnp.concatenate([init_observations, horizon_features], axis=-1)
        params = model.init(
            init_rng,
            init_observations,
            goals=init_goals,
            actions=example_batch[config['actions_key']],
            times=jnp.zeros_like(example_batch[config['actions_key']][..., :1]),
        )['params']
        network = TrainState.create(model, params, tx=optax.adam(config['lr']))
        config['action_dim'] = action_dim
        config['seed'] = seed
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))

##=========== TRAIN GOAL PROPOSER ===========##

example_batch = train_dataset.sample(1)
flow_agent = GCFlowGoalProposerAgent.create(example_batch, config)
jax.tree_util.tree_map(lambda x: x.shape, flow_agent.network.params)

flow_loss_history = []
velocity_mae_history = []
endpoint_mse_history = []

for step in range(1, config['num_train_steps'] + 1):
    batch = train_dataset.sample(config['batch_size'])
    flow_agent, info = flow_agent.update(batch)

    flow_loss_history.append(float(info['flow_loss']))
    velocity_mae_history.append(float(info['velocity_mae']))
    endpoint_mse_history.append(float(info['endpoint_mse']))

    if step == 1 or step % config['log_interval'] == 0:
        log_wandb(
            {
                'goal_proposer/flow_loss': float(info['flow_loss']),
                'goal_proposer/velocity_mae': float(info['velocity_mae']),
                'goal_proposer/endpoint_mse': float(info['endpoint_mse']),
            },
            step=step,
        )
        print(
            f"step={step:05d} flow_loss={flow_loss_history[-1]:.6f} "
            f"velocity_mae={velocity_mae_history[-1]:.6f} endpoint_mse={endpoint_mse_history[-1]:.6f}"
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
            action = dqc_agent.sample_actions(observations=ob, goals=subgoal, seed=action_rng, best_of_n_override=2)
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
