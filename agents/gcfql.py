import copy
from math import log
from typing import Any

import flax
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.plot_utils import get_block_i_pos_idxs

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue #, GoalProposalVectorField
from wrappers.datafuncs_utils import to_oracle_reps


class GCFQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow Q-learning (GCFQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        """Compute binary cross-entropy from logits against soft targets."""
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        return -(log_pred * target + log_not_pred * (1 - target))

    @jax.jit
    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"]

        if self.config['action_chunking']:
            next_actions = self.sample_actions(batch['next_observations'][..., -1, :], goals=batch['value_goals'], seed=sample_rng)
        else:
            next_actions = self.sample_actions(batch['next_observations'], goals=batch['value_goals'], seed=sample_rng)
        
        next_actions = jnp.clip(next_actions, -1, 1)

        if self.config['action_chunking']:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'][..., -1, :], goals=batch['value_goals'], actions=next_actions
            )
        else:
            next_qs = self.network.select('target_critic')(
                batch['next_observations'], goals=batch['value_goals'], actions=next_actions
            )

        if self.config['critic_loss_type'] == 'bce':
            next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'mean':
            next_q = next_qs.mean(axis=0)

        if self.config['action_chunking']:
            batch_rewards = batch['rewards'][..., -1]
            batch_masks = batch['masks'][..., -1]
        else:
            batch_rewards = batch['rewards']
            batch_masks = batch['masks']
        target_q = batch_rewards + self.config['discount'] * batch_masks * next_q
        # Clip target Q values to the valid range.
        if self.config['gc_negative']:
            target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
        else:
            target_q = jnp.clip(target_q, 0, 1)

        if self.config['critic_loss_type'] == 'squared':
            q = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch_actions, params=grad_params
            )
            critic_loss = jnp.square(q - target_q).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
            }

        elif self.config['critic_loss_type'] == 'bce':
            q_logit = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch_actions, params=grad_params
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

    @jax.jit
    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"]

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
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
    def goal_proposer_loss(self, batch, grad_params, rng):
        assert 'low_actor_goals' in batch, "low_actor_goals not in batch"
        # pred = self.network.select('goal_proposer')(
        #     observations=batch['observations'], params=grad_params
        # )
        # goal_loss = jnp.mean((pred - batch['low_actor_goals']) ** 2)
        # return goal_loss, {'goal_loss': goal_loss}
        # BC flow loss.
        batch_size, goal_dim = batch['low_actor_goals'].shape # TODO: visualize low_actor_goals
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, goal_dim))
        x_1 = batch['low_actor_goals']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        if self.config['goal_proposer_type'] == 'awr':
            assert False, "awr not supported currently"
            assert self.config["awr_invtemp"] > 0., "awr_invtemp should be > 0 for awr goal proposer"
            pred = self.network.select('goal_proposer')(
            observations=batch['observations'], goals=batch['value_goals'], actions=x_t, times=t, params=grad_params
            )
            
            v_s_g = self.network.select('value')(observations=batch['oracle_reps'], goals=batch['value_goals']).mean()
            v_s_sg = self.network.select('value')(observations=batch['oracle_reps'], goals=batch['low_actor_goals']).mean()
            v_sg_g = self.network.select('value')(observations=batch['low_actor_goals'], goals=batch['value_goals']).mean()

            adv = v_s_sg + (self.config['discount']**self.config["subgoal_steps"]) * v_sg_g - v_s_g
            weight = jnp.exp(adv * self.config["awr_invtemp"])
            weight = jnp.clip(weight, a_max=100.0)

            bc_flow_loss = jnp.mean(jnp.square(pred - vel) * weight)
        elif self.config['goal_proposer_type'] == 'actor-gc':
            pred = self.network.select('goal_proposer')(
            observations=batch['observations'], goals=batch['actor_goals'], actions=x_t, times=t, params=grad_params
            )
            bc_flow_loss = jnp.mean((pred - vel) ** 2)
        elif self.config['goal_proposer_type'] == 'value-gc':
            pred = self.network.select('goal_proposer')(
            observations=batch['observations'], goals=batch['value_goals'], actions=x_t, times=t, params=grad_params
            )
            bc_flow_loss = jnp.mean((pred - vel) ** 2)
        elif self.config['goal_proposer_type'] == 'low_actor_goals':
            pred = self.network.select('goal_proposer')(
            observations=batch['observations'], goals=batch['low_actor_goals'], actions=x_t, times=t, params=grad_params
            )
            bc_flow_loss = jnp.mean((pred - vel) ** 2)
        elif self.config['goal_proposer_type'] == 'default':
            pred = self.network.select('goal_proposer')(
            observations=batch['observations'], actions=x_t, times=t, params=grad_params
            )
            bc_flow_loss = jnp.mean((pred - vel) ** 2)
        else:
            assert False, "goal_proposer_type not recognized!"
    
        return bc_flow_loss, {'goal_proposer_loss': bc_flow_loss}

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @jax.jit
    # def value_loss(self, batch, grad_params, rng):
    #     pred = self.network.select('value')(
    #         observations=batch['oracle_reps'], goals=batch['value_goals'], params=grad_params
    #     )

    #     if self.config['use_policy_for_value']:
    #         policy_actions = self.sample_actions(batch['observations'], goals=batch['value_goals'], seed=rng)
    #     else:
    #         policy_actions = batch['actions']

    #     q_pred = self.network.select('target_critic')(
    #         batch['observations'], goals=batch['value_goals'], actions=policy_actions
    #     )

    #     if self.config['critic_loss_type'] == 'bce':
    #         q_pred = jax.nn.sigmoid(q_pred)
    #     if self.config['q_agg'] == 'min':
    #         q_pred = q_pred.min(axis=0)
    #     elif self.config['q_agg'] == 'mean':
    #         q_pred = q_pred.mean(axis=0)

    #     if self.config['critic_loss_type'] == 'squared':
    #         value_loss = jnp.square(pred - q_pred).mean()
    #     elif self.config['critic_loss_type'] == 'bce':
    #         value_loss = self.bce_loss(pred, q_pred).mean()
    #     else:
    #         assert False, 'set critic_loss_type to be a valid value!'

    #     pred_value = jax.nn.sigmoid(pred) if self.config['critic_loss_type'] == 'bce' else pred
    #     return value_loss, {'value_loss': value_loss, 'q_pred': q_pred.mean(), 'value_pred': pred_value.mean()}
    def value_loss(self, batch, grad_params, **kwargs):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    @partial(jax.jit, static_argnames=("env",))
    def total_loss(self, batch, grad_params, rng=None, env=None):
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

        loss = 0.0
        loss = critic_loss + actor_loss

        if self.config['train_value']:
            value_rng, rng = jax.random.split(rng)
            value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
            for k, v in value_info.items():
                info[f'value/{k}'] = v

            loss = loss + value_loss

        if self.config['train_goal_proposer']:
            goal_rng, rng = jax.random.split(rng)
            goal_propose_loss, goal_propose_info = self.goal_proposer_loss(batch, grad_params, goal_rng)

            for k, v in goal_propose_info.items():
                info[f'goal_proposer/{k}'] = v

            loss = loss + goal_propose_loss

        return loss, info

    # @jax.jit
    @partial(jax.jit, static_argnames=("module_name",))
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=("env",))
    def update(self, batch, env=None):
        """Update the agent and return a new agent with information dictionary."""

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, env=env)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        # self.target_update(new_network,)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def propose_goals(self, observations, goals, rng):
        goal_dim = goals.shape[-1]
        x = jax.random.normal(rng, (observations.shape[0], goal_dim))
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            if self.config['goal_proposer_type'] == 'default':
                vels = self.network.select('goal_proposer')(observations, actions=x, times=t)
                # still need to pass in the goals to make the function happy
            else:
                vels = self.network.select('goal_proposer')(observations, actions=x, goals=goals, times=t)
            x = x + vels / self.config['flow_steps']

        # need to clip goals?
        return x

    # @jax.jit
    @partial(jax.jit, static_argnames=("env",))
    def compute_dynamical_distance(self, states, goals, env):
        states = to_oracle_reps(states, env=env)
        v = self.network.select('value')(states, goals=goals)
        discount = self.config['discount']

        if self.config['critic_loss_type'] == 'bce':
            v = jax.nn.sigmoid(v)

        if self.config['gc_negative']:
            gamma_to_d = jnp.clip(1.0 + (1.0 - discount) * v, 1e-6, 1.0)
        else:
            gamma_to_d = jnp.clip(v, 1e-6, 1.0)

        return jnp.log(gamma_to_d) / jnp.log(discount)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0
    ):
        # TODO: no longer supports multiple goals

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        goals = jnp.repeat(goals[..., None, :], self.config['best_of_n'], axis=-2)
        actions = self.network.select(f'actor_onestep_flow')(observations, goals, noises)
        actions = jnp.clip(actions, -1, 1)
        
        q = self.network.select("critic")(observations, goals, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))
        
        return actions
    
    @jax.jit
    def best_of_n(
        self,
        observations,
        goals,
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
        mult_actions = self.sample_flow_actions(observations=mult_observations, goals=mult_goals, noises=noises)
        mult_actions = jnp.clip(mult_actions, -1, 1)

        qs = self.network.select('critic')(observations=mult_observations, goals=mult_goals, actions=mult_actions).mean(axis=0)
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
        if config['action_chunking']:
            ex_times = ex_actions[..., 0, :1]
        else:
            ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]

        # Define networks.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )

        if config["action_chunking"]:
            # full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
            full_actions = ex_actions.reshape(ex_actions.shape[0], -1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['layer_norm'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['layer_norm'],
        )

        if config['train_value'] or config['train_goal_proposer']:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )

        if config['train_goal_proposer']:
            goal_proposer_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=goal_dim, # TODO: double check that
                layer_norm=config['layer_norm'],
            )

        # action_dim = self.config['action_dim'] * \
        #                 (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        # if config['action_chunking']:
        #     ex_actions = jnp.zeros(action_dim * config['horizon_length'])[None] # TODO: properly do this? it's kind of hacky

        if config['train_goal_proposer']:
            if config['goal_proposer_type'] == 'default':
                network_info = dict(
                critic=(critic_def, (ex_observations, ex_goals, full_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, full_actions)),
                value=(value_def, (ex_goals, ex_goals)),
                actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, full_actions, ex_times)),
                actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, full_actions, None)),
                goal_proposer=(goal_proposer_def, (ex_observations, None, ex_goals, ex_times)) # TODO: check if this is reasonable?
            )
            else:
                network_info = dict(
                    critic=(critic_def, (ex_observations, ex_goals, full_actions)),
                    target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, full_actions)),
                    value=(value_def, (ex_goals, ex_goals)),
                    actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, full_actions, ex_times)),
                    actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, full_actions, None)),
                    goal_proposer=(goal_proposer_def, (ex_observations, ex_goals, ex_goals, ex_times)) # TODO: check if this is reasonable?
                )
        elif config['train_value']:
            network_info = dict(
                critic=(critic_def, (ex_observations, ex_goals, full_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, full_actions)),
                value=(value_def, (ex_goals, ex_goals)),
                actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, full_actions, ex_times)),
                actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, full_actions, None))
            )
        else:
            network_info = dict(
                critic=(critic_def, (ex_observations, ex_goals, full_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, full_actions)),
                actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, full_actions, ex_times)),
                actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, full_actions, None))
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
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            num_qs=10,  # Number of Q ensembles.
            q_agg='min',  # Aggregation function for target Q values.
            alpha=3.0,  # BC coefficient.
            critic_loss_type='bce',  # Value loss type ('squared' or 'bce').
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
            # actor_type='best-of-n',
            # num_actions=8,
            best_of_n=1,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            train_value=True,
            train_goal_proposer=False,
            subgoal_steps=25, # TODO: does this need to be changed?
            # value_loss_type='squared',
            awr_invtemp=0.0,
            goal_proposer_type='awr', # awr, default, value-gc, actor-gc,
            use_policy_for_value=False # TODO: delete this, eventually
        )
    )
    return config
