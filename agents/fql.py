import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import FQLActorVectorField, Value


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        if self.config['action_chunking']:
            batch_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        else:
            batch_actions = batch['actions']
        
        rng, sample_rng = jax.random.split(rng)

        if self.config['action_chunking']:
            next_actions = self.sample_actions(batch['next_observations'][..., -1, :], seed=sample_rng)
        else:
            next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        if self.config['action_chunking']:
            next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        else:
            next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'subtract-0.5x-std':
            mean_q = next_qs.mean(axis=0)
            std_q = next_qs.std(axis=0)
            next_q = mean_q - 0.5 * std_q
        else:
            next_q = next_qs.mean(axis=0)

        if self.config['action_chunking']:
            batch_rewards = batch['rewards'][..., -1]
            batch_masks = batch['masks'][..., -1]
        else:
            batch_rewards = batch['rewards']
            batch_masks = batch['masks']
        target_q = batch_rewards + (self.config['discount'] ** self.config['n_step']) * batch_masks * next_q

        if self.config['action_chunking']:
            q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        else:
            q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean() # Uses MSE loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config['action_chunking']:
            batch_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        else:
            batch_actions = batch['actions']
        
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions # batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))

        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch_actions) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
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

    # @jax.jit
    # def sample_actions(
    #     self,
    #     observations,
    #     seed=None,
    #     temperature=1.0,
    # ):
    #     """Sample actions from the one-step policy."""
    #     if self.config['actor_type'] == 'best-of-n':
    #         # return self.best_of_n(observations, self.config['num_actions'], seed)
    #         return self.best_of_n(observations, seed)
        
    #     elif self.config['actor_type'] == 'best-of-n-onestep':
    #         # return self.sample_actions_of_n(observations, self.config['num_actions'], seed)
    #         return self.sample_actions_of_n(observations, seed)
        
    #     action_seed, noise_seed = jax.random.split(seed)
    #     noises = jax.random.normal(
    #         action_seed,
    #         (
    #             *observations.shape[: -len(self.config['ob_dims'])],
    #             self.config['action_dim'],
    #         ),
    #     )
    #     actions = self.network.select('actor_onestep_flow')(observations, noises)
    #     actions = jnp.clip(actions, -1, 1)
    #     return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
    ):
        action_dim = self.config['action_dim'] * (
            self.config['horizon_length'] if self.config['action_chunking'] else 1
        )
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        actions = self.network.select(f'actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))
        
        return actions

    @jax.jit
    def sample_actions_of_n(
        self,
        observations,
        # num_actions,
        seed=None,
        temperature=1.0,
    ):
        num_actions = self.config['actor_num_samples']
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        mult_observations = jnp.repeat(observations[..., None, :], num_actions, axis=-2)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
                num_actions
            ),
        )
        mult_actions = self.network.select('actor_onestep_flow')(observations, noises) # TODO: does this need to be changed to mult_observations?
        mult_actions = jnp.clip(mult_actions, -1, 1)

        qs = self.network.select('critic')(mult_observations, actions=mult_actions).mean(axis=0)
        idx = jnp.argmax(qs, axis=-1)
        actions = mult_actions[idx]

        return actions
    
    @jax.jit
    def best_of_n(
        self,
        observations,
        # num_actions,
        seed=None
    ):
        num_actions = self.config['actor_num_samples']
        action_seed, noise_seed = jax.random.split(seed)
        mult_observations = jnp.repeat(observations[..., None, :], num_actions, axis=-2)

        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                num_actions,
                self.config['action_dim']
            ),
        )
        mult_actions = self.compute_flow_actions(mult_observations, noises)
        mult_actions = jnp.clip(mult_actions, -1, 1)

        qs = self.network.select('critic')(mult_observations, actions=mult_actions).mean(axis=0)
        idx = jnp.argmax(qs, axis=-1)
        # actions = mult_actions[idx]
        bshape = idx.shape
        idx = idx.reshape(-1)
        bsize = len(idx)

        actions = jnp.reshape(mult_actions, (-1, self.config["actor_num_samples"], self.config['action_dim']))[jnp.arange(bsize), idx, :].reshape(
                bshape + (self.config['action_dim'],))

        return actions
        

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['action_chunking']:
            ex_times = ex_actions[..., 0, :1]
        else:
            ex_times = ex_actions[..., :1]

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        if config['action_chunking']:
            full_actions = ex_actions.reshape(ex_actions.shape[0], -1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'], # Made this flexible.
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = FQLActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = FQLActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
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
            agent_name='fql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            num_qs=10,

            best_of_n=1,
            n_step=1,
            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set
            action_chunking=False,                                      # Use Q-chunking or just n-step return
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
