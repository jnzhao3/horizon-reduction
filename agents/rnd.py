import functools
from functools import partial
from typing import Any, Callable, Optional, Sequence, Type

import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

import flax.linen as nn


import ml_collections

def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: float = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


class RND(struct.PyTreeNode):
    rng: Any
    net: TrainState
    init_net: TrainState
    frozen_net: TrainState
    coeff: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed,
        observation_example,
        action_example,
        config,
    ):
        
        lr = config.lr
        coeff = config.coeff
        hidden_dims = config.hidden_dims
        
        rng = jax.random.PRNGKey(seed)
        rng, key1, key2 = jax.random.split(rng, 3)

        net_def = MLP(hidden_dims=hidden_dims, activate_final=False)
        params = FrozenDict(net_def.init(key1, observation_example, action_example)["params"])
        net = TrainState.create(
            apply_fn=net_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
        )
        frozen_params = FrozenDict(net_def.init(key2, observation_example, action_example)["params"])
        frozen_net = TrainState.create(
            apply_fn=net_def.apply,
            params=frozen_params,
            tx=optax.adam(learning_rate=lr),
        )
        return cls(
            rng=rng,
            init_net=net,
            net=net,
            frozen_net=frozen_net,
            coeff=coeff,
        )

    @jax.jit
    def reset(self):
        return self.replace(net=self.init_net)

    @jax.jit
    def update(self, batch):
        obs_key = 'oracle_reps' if 'oracle_reps' in batch else 'observations'
        observations = batch[obs_key]
        actions = batch['actions']
        
        def loss_fn(params):
            feats = self.net.apply_fn(
                {"params": params}, observations, actions
            )
            frozen_feats = self.frozen_net.apply_fn(
                {"params": self.frozen_net.params},
                observations,
                actions,
            )
            loss = ((feats - frozen_feats) ** 2.0).mean()
            return loss, {"rnd_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.net.params)
        net = self.net.apply_gradients(grads=grads)

        return self.replace(net=net), info

    @partial(jax.jit, static_argnames=("stats",))
    def get_reward(self, observations, actions, stats=False):
        feats = self.net.apply_fn({"params": self.net.params}, observations, actions)
        frozen_feats = self.net.apply_fn(
            {"params": self.frozen_net.params}, observations, actions
        )
        reward = jnp.mean((feats - frozen_feats) ** 2.0, axis=-1) * self.coeff
        if stats:
            stats = {
                "mean": jnp.mean(reward),
                "std": jnp.std(reward),
                "max": jnp.max(reward),
                "min": jnp.min(reward),
            }
            return reward, stats
        return reward

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rnd',
            lr=3e-4,
            coeff=1.0,
            hidden_dims=(512, 512, 512),
        )
    )
    return config
