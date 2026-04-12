import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

        (self.terminal_locs,) = np.nonzero(self['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)
        
    # @jax.jit
    # def

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        rewards = np.zeros(data['rewards'].shape + (sequence_length,), dtype=float)
        masks = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        valid = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        next_observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        actions = np.zeros(data['actions'].shape[:-1] + (sequence_length, data['actions'].shape[-1]), dtype=float)
        terminals = np.zeros(data['terminals'].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            cur_idxs = idxs + i

            if i == 0:
                rewards[..., 0] = self['rewards'][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = (1.0 - terminals[..., i - 1])
                rewards[..., i] = rewards[..., i - 1] + (self['rewards'][cur_idxs] * (discount ** i) * valid[..., i])
                masks[..., i] = np.minimum(masks[..., i-1], self["masks"][cur_idxs]) * valid[..., i] + masks[..., i-1] * (1. - valid[..., i])
                terminals[..., i] = np.maximum(terminals[..., i-1], self["terminals"][cur_idxs])
            
            actions[..., i, :] = self['actions'][cur_idxs]
            next_observations[..., i, :] = self['next_observations'][cur_idxs] * valid[..., i:i+1] + next_observations[..., i-1, :] * (1. - valid[..., i:i+1])
            observations[..., i, :] = self['observations'][cur_idxs]
            
        return dict(
            observations=data['observations'].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        # self.size = max(self.pointer, self.size)
        self.size = min(self.size + 1, self.max_size) # bug fix

    def add_transitions(self, transitions):

        batch_size = jax.tree_util.tree_leaves(transitions)[0].shape[0]

        for i in range(batch_size):
            idx = (self.pointer + i) % self.max_size

            def set_idx(buffer, new_element):
                buffer[idx] = new_element

            # Extract i-th transition across the PyTree and write it at buffer[idx]
            single_transition = jax.tree_util.tree_map(lambda x: x[i], transitions)
            jax.tree_util.tree_map(set_idx, self._dict, single_transition)

        self.pointer = (self.pointer + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def combine_with(self, second_dataset):
        # assert type(second_dataset) == type(self)
        self.max_size += second_dataset.size

        transitions = second_dataset.dataset.unfreeze()
        batch_size = transitions['observations'].shape[0]
        transitions = second_dataset.sample(batch_size, idxs=np.arange(batch_size))

        for k, v in self._dict.items():
            if v.ndim >= 2:
                self._dict[k] = np.concatenate([v, np.zeros((batch_size, self._dict[k].shape[1]))])
            else:
                self._dict[k] = np.concatenate([v, np.zeros((batch_size))])

        for i in range(batch_size):
            idx = (self.pointer + i) % self.max_size

            def set_idx(buffer, new_element):
                buffer[idx] = new_element

            single_transition = jax.tree_util.tree_map(lambda x: x[0], transitions)
            jax.tree_util.tree_map(set_idx, self._dict, single_transition)
        # self.add_transitions(transitions)

        self.pointer = (self.pointer + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
        
@partial(jax.jit, static_argnames=('valids',))
def jax_get_random_idxs(batch_size, rng, size, valids=None):
    """Return `batch_size` random indices."""
    if valids is not None:
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, valids.shape[0])
        return key, valids[idx]
    else:
        key, sub = jax.random.split(key)
        return key, jax.random.randint(sub, (batch_size,), 0, size)
    
@jax.jit
def jax_sample_batch(dataset, idxs):
    result = jax.tree_util.tree_map(lambda arr: arr[idxs], dataset)
    return result

@jax.jit
def jax_sample_next_observations(observations, size, idxs):
    return observations[jnp.minimum(idxs + 1, size - 1)]

@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        # assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        actor_subgoal_steps = (
            self.config['subgoal_steps'] # TODO: check this value
            # if self.config.get('actor_subgoal_steps') is None
            # else self.config['actor_subgoal_steps']
        ) # TODO: i could also remove this and just use HGCDataset

        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
            batch['actor_goals'] = self.dataset['oracle_reps'][actor_goal_idxs]
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)
        # batch['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)
        if 'oracle_reps' in self.dataset:
            batch['low_actor_goals'] = self.dataset['oracle_reps'][low_actor_goal_idxs]
        else:
            batch['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)

        return batch

    def sample_sequence(self, batch_size: int, sequence_length: int, discount: float):
        idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        actor_subgoal_steps = self.config['subgoal_steps']
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

        seq_offsets = np.arange(sequence_length)
        seq_idxs = idxs[:, None] + seq_offsets[None, :]
        clipped_seq_idxs = np.minimum(seq_idxs, final_state_idxs[:, None])

        # valid steps
        valid = (seq_idxs <= final_state_idxs[:, None]).astype(float)

        # actions (UNFLATTENED)
        actions = self.dataset['actions'][clipped_seq_idxs]
        actions = actions * valid[..., None].astype(actions.dtype)

        # next observations (sequence)
        if 'next_observations' in self.dataset:
            next_observations = self.dataset['next_observations'][clipped_seq_idxs]
        else:
            next_obs_idxs = np.minimum(clipped_seq_idxs + 1, self.size - 1)
            next_observations = self.get_observations(next_obs_idxs.reshape(-1)).reshape(
                batch_size, sequence_length, -1
            )

        # freeze next_obs after invalid
        for i in range(1, sequence_length):
            next_observations[:, i, :] = (
                next_observations[:, i, :] * valid[:, i:i+1]
                + next_observations[:, i - 1, :] * (1.0 - valid[:, i:i+1])
            )

        # terminals (cumulative)
        step_terminals = self.dataset['terminals'][clipped_seq_idxs] * valid
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        terminals[:, 0] = step_terminals[:, 0]
        for i in range(1, sequence_length):
            terminals[:, i] = np.maximum(terminals[:, i - 1], step_terminals[:, i])

        # goal success
        success_offsets = value_goal_idxs - idxs
        successes = (success_offsets[:, None] == seq_offsets[None, :]).astype(float)

        # prefix rewards
        rewards = np.zeros((batch_size, sequence_length), dtype=float)

        for i in range(sequence_length):
            valid_i = valid[:, i]

            if np.isclose(discount, 1.0):
                pos_i = successes[:, i]
                neg_i = -valid_i
            else:
                pos_i = successes[:, i] * (discount ** i)
                neg_i = -(discount ** i) * valid_i

            reward_i = neg_i if self.config['gc_negative'] else pos_i

            if i == 0:
                rewards[:, 0] = reward_i
            else:
                still_alive = 1.0 - terminals[:, i - 1]
                rewards[:, i] = rewards[:, i - 1] + reward_i * still_alive

        # masks = 1 - successes (per step)
        masks = 1.0 - successes

        batch_out = dict(
            observations=batch['observations'].copy(),
            actions=actions,
            rewards=rewards,
            masks=masks,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )

        if 'oracle_reps' in self.dataset:
            batch_out['oracle_reps'] = batch['oracle_reps'].copy()
            batch_out['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
            batch_out['actor_goals'] = self.dataset['oracle_reps'][actor_goal_idxs]
            batch_out['low_actor_goals'] = self.dataset['oracle_reps'][low_actor_goal_idxs]
        else:
            batch_out['value_goals'] = self.get_observations(value_goal_idxs)
            batch_out['actor_goals'] = self.get_observations(actor_goal_idxs)
            batch_out['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)

        return batch_out

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support hierarchical goal-conditioned RL. It reads the following additional key from
    the config:
    - subgoal_steps (optional: value_subgoal_steps, actor_subgoal_steps): Subgoal steps. It is also possible to specify
        `value_subgoal_steps` and `actor_subgoal_steps` separately.
    - low_discount: If specified, return low-level value goals as well.
    """

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, subgoal_steps):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # Clip to the end of the trajectory.
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def get_high_actions(self, target_idxs, cur_idxs):
        if 'oracle_reps' in self.dataset:
            return self.dataset['oracle_reps'][target_idxs]
        else:
            return self.get_observations(target_idxs)

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        value_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('value_subgoal_steps') is None
            else self.config['value_subgoal_steps']
        )
        high_value_next_idxs, high_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_value_reps'] = self.dataset['oracle_reps'][idxs]
            batch['high_value_goals'] = self.dataset['oracle_reps'][high_value_goal_idxs]
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        else:
            batch['high_value_reps'] = batch['observations']
            batch['high_value_goals'] = self.get_observations(high_value_goal_idxs)
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        batch['high_value_offsets'] = high_value_goal_idxs - idxs

        high_value_successes = (high_value_subgoal_steps < value_subgoal_steps).astype(float)
        batch['high_value_subgoal_steps'] = high_value_subgoal_steps
        batch['high_value_masks'] = 1.0 - high_value_successes
        if self.config['gc_negative']:
            batch['high_value_rewards'] = -(1 - self.config['discount'] ** high_value_subgoal_steps) / (
                1 - self.config['discount']
            )
        else:
            batch['high_value_rewards'] = (self.config['discount'] ** high_value_subgoal_steps) * high_value_successes

        # Sample low-level value goals (if requested).
        if 'low_discount' in self.config:
            low_value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                geom_sample=True,
                discount=self.config['low_discount'],
            )

            if 'oracle_reps' in self.dataset:
                batch['low_value_goals'] = self.dataset['oracle_reps'][low_value_goal_idxs]
            else:
                batch['low_value_goals'] = self.get_observations(low_value_goal_idxs)
            successes = (idxs == low_value_goal_idxs).astype(float)
            batch['low_value_masks'] = 1.0 - successes
            batch['low_value_rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # One-step information.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample high-level actor goals.
        high_actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )
        actor_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('actor_subgoal_steps') is None
            else self.config['actor_subgoal_steps']
        )
        high_actor_next_idxs, high_actor_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            actor_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_actor_goals'] = self.dataset['oracle_reps'][high_actor_goal_idxs]
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)
        else:
            batch['high_actor_goals'] = self.get_observations(high_actor_goal_idxs)
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)

        # Compute low-level actor goals.
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

        batch['low_actor_goals'] = self.get_high_actions(low_actor_goal_idxs, idxs)

        return batch
    
class WeightedReplayBufferWrapper:
    """Wrapper for a replay buffer with weighted sampling.

    This class wraps a replay buffer and provides a method to sample transitions with weights.
    """

    def __init__(self, replay_buffer, default_weight=1.0):
        self.replay_buffer = replay_buffer
        self.weights = np.full(len(replay_buffer), default_weight, dtype=np.float32)

    def reweight(self, fn=lambda transition: 0.0):
        """Reweight the transitions in the replay buffer using the given function.

        Args:
            fn: Function that takes a transition and returns a weight.
        """
        for i in range(len(self.replay_buffer)):
            transition = self.replay_buffer.get_subset([i])
            self.weights[i] = fn(transition)

    def sample(self, batch_size):
        """Sample a batch of transitions with weights."""
        idxs = np.random.choice(
            len(self.replay_buffer), size=batch_size, p=self.weights / np.sum(self.weights)
        )
        return self.replay_buffer.get_subset(idxs)
    
    def add_transition(self, transition, weight=1.0):
        """Add a transition to the replay buffer with a weight."""
        self.replay_buffer.add_transition(transition)
        self.weights = np.append(self.weights, weight)

    def clear(self):
        """Clear the replay buffer and weights."""
        self.replay_buffer.clear()
        self.weights = np.array([], dtype=np.float32)

    def get_subset(self, idxs):
        """Return a subset of the replay buffer given the indices."""
        return self.replay_buffer.get_subset(idxs)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return self.replay_buffer.get_random_idxs(num_idxs)
    
class SimpleWeightedReplayBufferWrapper(ReplayBuffer):
    """Wrapper for a replay buffer with weighted sampling.

    This class wraps a replay buffer and provides a method to sample transitions with weights.
    """
    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size, new_to_old_ratio=0.5):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        dataset.old_size = dataset.size
        dataset.new_to_old_ratio = new_to_old_ratio
        return dataset
    
    def get_random_idxs_old(self, num_idxs):
        """Return `num_idxs` random indices from the old transitions."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.old_size, size=num_idxs)
        
    def get_random_idxs_new(self, num_idxs):
        """Return `num_idxs` random indices from the new transitions."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size - self.old_size, size=num_idxs) + self.old_size
    
    def sample(self, batch_size, special_ratio=None):
        """Sample a batch of transitions with weights."""
        if special_ratio is not None:
            # If a special ratio is provided, use it to determine the number of new and old transitions.
            num_new = int(batch_size * special_ratio)
            num_old = batch_size - num_new
        else:
            num_new = int(batch_size * self.new_to_old_ratio)
            num_old = batch_size - num_new

        # new_idxs = self.get_random_idxs(num_new)
        # old_idxs = self.get_random_idxs(num_old)
        new_idxs = self.get_random_idxs_new(num_new)
        old_idxs = self.get_random_idxs_old(num_old)

        new_batch = self.get_subset(new_idxs)
        old_batch = self.get_subset(old_idxs)

        return jax.tree_util.tree_map(
            lambda new, old: np.concatenate([new, old], axis=0), new_batch, old_batch
        )

    def get_num_valids(self):
        """Return the number of valid transitions in the replay buffer."""
        if 'valids' in self._dict:
            return np.sum(self._dict['valids'])
        else:
            return self.size

# @dataclasses.dataclass
# class GCDataset:
#     """Dataset class for goal-conditioned RL.

#     This class provides a method to sample a batch of transitions with goals from the
#     dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

#     It reads the following keys from the config:
#     - discount: Discount factor for geometric sampling.
#     - value_p_curgoal: Probability of using the current state as the value goal.
#     - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
#     - value_p_randomgoal: Probability of using a random state as the value goal.
#     - value_geom_sample: Whether to use geometric sampling for future value goals.
    
#     Attributes:
#         dataset: Dataset object.
#         config: Configuration dictionary.
#     """

#     dataset: Dataset
#     config: Any

#     def __post_init__(self):
#         self.size = self.dataset.size

#         # Pre-compute trajectory boundaries.
#         (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
#         self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
#         assert self.terminal_locs[-1] == self.size - 1

#         # Assert probabilities sum to 1.
#         assert np.isclose(
#             self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
#         )

#     def sample(self, batch_size: int, idxs=None, evaluation=False):
#         """Sample a batch of transitions with goals.

#         This method samples a batch of transitions with goals from the dataset. They are
#         stored in the key 'value_goals', respectively. It also computes the 'rewards' and 'masks'
#         based on the indices of the goals.

#         Args:
#             batch_size: Batch size.
#             idxs: Indices of the transitions to sample. If None, random indices are sampled.
#             evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
#         """
#         if idxs is None:
#             idxs = self.dataset.get_random_idxs(batch_size)

#         batch = self.dataset.sample(batch_size, idxs)
        
#         value_goal_idxs = self.sample_goals(
#             idxs,
#             self.config['value_p_curgoal'],
#             self.config['value_p_trajgoal'],
#             self.config['value_p_randomgoal'],
#             self.config['value_geom_sample'],
#         )
        
#         if 'oracle_reps' in self.dataset:
#             batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
#         else:
#             batch['value_goals'] = self.get_observations(value_goal_idxs)
#         successes = (idxs == value_goal_idxs).astype(float)
#         batch['masks'] = 1.0 - successes
#         batch['rewards'] = successes

#         actor_subgoal_steps = (
#             self.config['subgoal_steps']
#             if self.config.get('actor_subgoal_steps') is None
#             else self.config['actor_subgoal_steps']
#         )
#         low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

#         if 'oracle_reps' in self.dataset:
#             batch['low_actor_goals'] = self.dataset['oracle_reps'][low_actor_goal_idxs]
#         else:
#             batch['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)

#         return batch

#     def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
#         """Sample goals for the given indices."""
#         batch_size = len(idxs)
#         if discount is None:
#             discount = self.config['discount']

#         # Random goals.
#         random_goal_idxs = self.dataset.get_random_idxs(batch_size)

#         # Goals from the same trajectory (excluding the current state, unless it is the final state).
#         final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
#         if geom_sample:
#             # Geometric sampling.
#             offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
#             traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
#         else:
#             # Uniform sampling.
#             distances = np.random.rand(batch_size)  # in [0, 1)
#             traj_goal_idxs = np.round(
#                 (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
#             ).astype(int)
#         if p_curgoal == 1.0:
#             goal_idxs = idxs
#         else:
#             goal_idxs = np.where(
#                 np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
#             )

#             # Goals at the current state.
#             goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

#         return goal_idxs
    
#     def get_observations(self, idxs):
#         """Return the observations for the given indices."""
#         return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])

@dataclasses.dataclass
class CGCDataset(GCDataset):
    """Dataset class for action-chunked goal-conditioned RL.

    This class extends GCDataset to support action chunks. It reads the following additional key from the config:
    - backup_horizon: Subgoal steps.
    """

    def __post_init__(self):
        super().__post_init__()
        # Set valid_idxs for action chunks.
        cur_idx = 0
        valid_idxs = []
        
        for terminal_idx in self.terminal_locs:
            valid_idxs.append(np.arange(cur_idx, terminal_idx + 1 - self.config['backup_horizon']))
            cur_idx = terminal_idx + 1
        self.dataset.valid_idxs = np.concatenate(valid_idxs)

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, backup_horizon):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        backup_horizon = np.full(batch_size, backup_horizon)

        # Clip to the end of the trajectory.
        backup_horizon = np.minimum(backup_horizon, final_state_idxs - idxs)

        # assert np.all(backup_horizon == self.config['backup_horizon'])

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < backup_horizon)
        backup_horizon = np.where(should_clip, diff_idxs, backup_horizon)

        return idxs + backup_horizon, backup_horizon

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        value_backup_horizon = self.config['backup_horizon'] if self.config.get('value_backup_horizon') is None else self.config['value_backup_horizon']
        high_value_next_idxs, high_value_backup_horizon = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_backup_horizon,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_value_reps'] = self.dataset['oracle_reps'][idxs]
            batch['high_value_goals'] = self.dataset['oracle_reps'][high_value_goal_idxs]
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        else:
            batch['high_value_reps'] = batch['observations']
            batch['high_value_goals'] = self.get_observations(high_value_goal_idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        
        batch['high_value_offsets'] = high_value_goal_idxs - idxs
        chunk_offsets = np.arange(self.config['backup_horizon'])
        chunk_idxs = np.minimum(idxs[:, None] + chunk_offsets, final_state_idxs[:, None])
        batch["valids"] = (idxs[:, None] + chunk_offsets <= final_state_idxs[:, None]).astype(float)
        batch['high_value_action_chunks'] = self.dataset['actions'][chunk_idxs].reshape(batch_size, -1)

        high_value_successes = (high_value_backup_horizon < value_backup_horizon).astype(float)
        batch['high_value_backup_horizon'] = high_value_backup_horizon
        batch['high_value_masks'] = 1.0 - high_value_successes
        batch['high_value_rewards'] = (self.config['discount'] ** high_value_backup_horizon) * high_value_successes

        # One-step info.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes

        actor_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('actor_subgoal_steps') is None
            else self.config['actor_subgoal_steps']
        )
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

        if 'oracle_reps' in self.dataset:
            batch['low_actor_goals'] = self.dataset['oracle_reps'][low_actor_goal_idxs]
        else:
            batch['low_actor_goals'] = self.get_observations(low_actor_goal_idxs)

        return batch
