import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue


class GCFQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow Q-learning (GCFQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], goals=batch['value_goals'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'], goals=batch['value_goals'], actions=next_actions
        )
        if self.config['value_loss_type'] == 'bce':
            next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'mean':
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        # Clip target Q values to the valid range.
        if self.config['gc_negative']:
            target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
        else:
            target_q = jnp.clip(target_q, 0, 1)

        if self.config['value_loss_type'] == 'squared':
            q = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch['actions'], params=grad_params
            )
            critic_loss = jnp.square(q - target_q).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
            }

        elif self.config['value_loss_type'] == 'bce':
            q_logit = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch['actions'], params=grad_params
            )
            q = jax.nn.sigmoid(q_logit)
            log_q = jax.nn.log_sigmoid(q_logit)
            log_not_q = jax.nn.log_sigmoid(-q_logit)
            critic_loss = -(log_q * target_q + log_not_q * (1 - target_q)).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
                'q_logit_mean': q_logit.mean(),
                'q_logit_max': q_logit.max(),
                'q_logit_min': q_logit.min(),
            }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], goals=batch['actor_goals'], actions=x_t, times=t, params=grad_params
        )
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.sample_flow_actions(batch['observations'], goals=batch['actor_goals'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], goals=batch['actor_goals'], actions=noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], goals=batch['actor_goals'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        # Make it scale-invariant.
        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'vel': jnp.linalg.norm(vel, axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""

        if self.config['actor_type'] == 'best-of-n':
            # return self.best_of_n(observations, self.config['num_actions'], seed)
            return self.best_of_n(observations=observations, goals=goals, seed=seed)
        
        mult_goals = goals.ndim > 1 and observations.ndim == 1

        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )

        if mult_goals:
            observations = jnp.repeat(observations[None], goals.shape[0], axis=0)
            actions = jnp.repeat(actions[None], goals.shape[0], axis=0)
            
        actions = self.network.select('actor_onestep_flow')(observations=observations, goals=goals, actions=actions)

        if mult_goals:
            actions = actions.mean(axis=0)

        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def best_of_n(
        self,
        observations,
        goals,
        # num_actions,
        seed=None
    ):
        num_actions = self.config['num_actions']
        action_seed, noise_seed = jax.random.split(seed)
        mult_observations = jnp.repeat(observations[..., None, :], num_actions, axis=-2)
        mult_goals = jnp.repeat(goals[..., None, :], num_actions, axis=-2)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                num_actions,
                self.config['action_dim']
            ),
        )
        mult_actions = self.sample_flow_actions(observations=mult_observations, noises=noises, goals=mult_goals)
        mult_actions = jnp.clip(mult_actions, -1, 1)

        qs = self.network.select('critic')(mult_observations, goals=mult_goals, actions=mult_actions).mean(axis=0)
        idx = jnp.argmax(qs, axis=-1)
        # actions = mult_actions[idx]
        bshape = idx.shape
        idx = idx.reshape(-1)
        bsize = len(idx)

        actions = jnp.reshape(mult_actions, (-1, self.config["num_actions"], self.config['action_dim']))[jnp.arange(bsize), idx, :].reshape(
                bshape + (self.config['action_dim'],))

        return actions

    @jax.jit
    def sample_emaq_actions(
        self,
        observations,
        goals=None,
        rng=None,
        # seed=None,
        # temperature=1.0,
    ):
        batch_size = goals.shape[0]
        action_dim = self.config['action_dim']
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actions = noises

        observations = jnp.repeat(observations[None], goals.shape[0], axis=0)
        # actions = jnp.repeat(actions[None], goals.shape[0], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, goals=goals, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        qs = self.network.select('critic')(observations=observations, goals=goals, actions=actions).mean(axis=0)
        action_idx = jnp.argmax(qs)
        return actions[action_idx]
    
    @jax.jit
    def sample_guidance_actions(
        self,
        observations,
        goals=None,
        rng=None,
    ):
        # batch_size = goals.shape[0]
        action_dim = self.config['action_dim']
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (action_dim))
        actions = noises

        observations = jnp.repeat(observations[None], goals.shape[0], axis=0)
        actions = jnp.repeat(actions[None], goals.shape[0], axis=0)

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, goals=goals, actions=actions, times=t)
            vels = vels.mean(axis=0)
            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        action = actions[0]
        # action = actions.mean(axis=0) # TODO: weighted mean, instead of mean?
        # action = actions
        return action

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        noises,
        goals=None,
    ):
        """Sample actions from the BC flow policy."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, goals=goals, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['actor_goals']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define networks.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, ex_actions, None)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcfql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            num_qs=2,  # Number of Q ensembles.
            q_agg='min',  # Aggregation function for target Q values.
            alpha=3.0,  # BC coefficient.
            value_loss_type='bce',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            actor_type='best-of-n',
            num_actions=8,
        )
    )
    return config
