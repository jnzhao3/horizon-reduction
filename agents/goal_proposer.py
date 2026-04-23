import glob
import os
import re
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.datasets import CGCDataset
from utils.flax_utils import TrainState, nonpytree_field, restore_agent
from utils.networks import ActorVectorField

class GoalProposerCGCDataset(CGCDataset):
    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        if not self.config.get('horizon_conditioned', False):
            return super().sample(batch_size, idxs=idxs, evaluation=evaluation)

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


def make_goal_proposer_config(
    name,
    *,
    observation_conditioned,
    horizon_conditioned,
    seed_offset=0,
    base_config=None,
):
    base_config = {} if base_config is None else dict(base_config)
    proposer_config = dict(base_config)
    proposer_config.update(
        output_dir=f'checkpoints/gc_flow_goal_proposer/{name}',
        num_train_steps=3000000,
        batch_size=256,
        log_interval=1000,
        seed=int(base_config.get('seed', 0)) + int(seed_offset),
        observations_key=base_config.get('observations_key', 'observations'),
        goal_key=base_config.get('goal_key', 'actor_goals'),
        actions_key='low_actor_goals',
        hidden_dims=tuple(base_config.get('hidden_dims', (256, 256, 256))),
        layer_norm=bool(base_config.get('layer_norm', True)),
        lr=float(base_config.get('lr', 3e-4)),
        flow_steps=int(base_config.get('flow_steps', 10)),
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=True,
        gc_negative=False,
        discount=float(base_config.get('discount', 0.995)),
        backup_horizon=int(base_config.get('backup_horizon', 25)),
        goal_conditioned=False,
        observation_conditioned=bool(observation_conditioned),
        horizon_conditioned=bool(horizon_conditioned),
        horizon_key='horizons',
        min_horizon_steps=1,
        max_horizon_steps=100,
        subgoal_steps=100,
        horizon_scale=float(100),
    )
    return proposer_config


def latest_checkpoint_step(checkpoint_dir):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, 'params_*.pkl'))
    if not checkpoint_paths:
        raise FileNotFoundError(f'No params_*.pkl checkpoints found in {checkpoint_dir}')

    def checkpoint_step(path):
        match = re.search(r'params_(\d+)\.pkl$', os.path.basename(path))
        if match is None:
            raise ValueError(f'Unexpected checkpoint filename: {path}')
        return int(match.group(1))

    return max(checkpoint_step(path) for path in checkpoint_paths)


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

    def sample_actions(self, observations=None, goals=None, rng=None, seed=None, num_samples=None, horizons=None):
        if rng is None:
            rng = self.rng if seed is None else seed

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


def restore_gc_flow_goal_proposer(name, example_batch, config, checkpoint_root='checkpoints/gc_flow_goal_proposer'):
    checkpoint_dir = os.path.join(checkpoint_root, name)
    checkpoint_step = latest_checkpoint_step(checkpoint_dir)
    agent = GCFlowGoalProposerAgent.create(example_batch, config)
    return restore_agent(agent, checkpoint_dir, checkpoint_step), checkpoint_step
