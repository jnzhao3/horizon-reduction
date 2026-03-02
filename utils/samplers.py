from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
# abstract base class
from abc import ABC, abstractmethod
from agents import agents
import json
import pickle
from tqdm import tqdm
import flax
import optax
from flax import struct
from flax.training.train_state import TrainState
import os
from utils.networks import RND
from utils.plot_utils import get_block_i_pos_idxs
# from agents.rnd import RND

def make_sampler(sampler_type, constructor, sampler_params, env_name, train_dataset, goal_xy, discount, agent=None):
    try:
        sampler_constructor = eval(f"{sampler_type}.{constructor}")
    except KeyError:
        raise ValueError(f"Sampler type '{sampler_type}' not found in the configuration file.")
    except NameError:
        raise ValueError(f"Sampler type '{sampler_type}' is not defined. Please ensure it is implemented.")
    except Exception as e:
        raise ValueError(f"An error occurred while trying to create the sampler: {e}")

    try:
        # sampler = sampler_class(**config.get("sampler_params", {}))
        sampler = sampler_constructor(
            **sampler_params,
            env_name=env_name,
            train_dataset=train_dataset,
            goal_xy=goal_xy,
            discount=discount,
            value_agent=agent
        )
    except TypeError as e:
        raise ValueError(f"Error initializing sampler '{sampler_type}': {e}")
    except Exception as e:
        raise ValueError(f"An error occurred while initializing the sampler: {e}")
    return sampler

# @partial(jax.jit, static_argnames=('subset',))
def precompute_gciql_values(value_agent, subset):
    pos = subset[:, :2]
    N = pos.shape[0]

    value_net = value_agent.network.select('value')
    pairwise_values = jnp.zeros((N, N))

    step_size = 500
    for i in tqdm(range(0, N, step_size)):
        row_end = min(i + step_size, N)
        subset_i = subset[i:row_end]
        for j in tqdm(range(0, N, step_size)):
            endp = min(j + step_size, N)
            subset_j = subset[j:endp]
            grid_values = compute_values(subset_i, subset_j, value_net)
            pairwise_values = pairwise_values.at[i:row_end, j:endp].set(grid_values)
    # i_prep = jnp.repeat(subset, subset.shape[0], axis=0)
    # j_prep = jnp.tile(subset, (subset.shape[0], 1))
    # flattened_values = value_net(i_prep, j_prep)
    # pairwise_values = jnp.reshape(flattened_values, (subset.shape[0], -1))
    return pairwise_values, pos

@jax.jit
def compute_values(subset_i, subset_j, value_agent):
    value_net = value_agent.network.select('value')
    subset_i_prep = jnp.repeat(subset_i, subset_j.shape[0], axis=0)
    subset_j_prep = jnp.tile(subset_j, (subset_i.shape[0], 1))
    flattened_values = value_net(subset_i_prep, subset_j_prep)
    grid_values = jnp.reshape(flattened_values, (subset_i.shape[0], -1))
    return grid_values

def to_oracle_rep(obs, env):
    env_name = env.spec.id
    if 'maze' in env_name:
        # return obs[:2]
        return obs[:, :2]
    elif 'cube' in env_name:
        num_cubes = env.unwrapped.task_infos[0]['init_xyzs'].shape[0]

        ob = []
        for i in range(num_cubes):
            pos = get_block_i_pos_idxs(i, num_cubes)
            # ob.append(obs[pos])
            ob.append(obs[:, pos])
            # ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)
        return jnp.concatenate(jnp.array(ob), axis=-1)
    else:
        assert False, 'not implemented'


def reached_subgoal(env_name, subgoal, observation):
    if 'humanoidmaze' in env_name:
        return np.linalg.norm(subgoal[:2] - observation[:2]) <= 0.5
    if 'cube-triple' in env_name:
        if subgoal.shape[0] == 9: # oraclerep
            return (np.linalg.norm(subgoal[:3] - observation[:3]) <= 0.04 and
                    np.linalg.norm(subgoal[3:6] - observation[3:6]) <= 0.04 and
                    np.linalg.norm(subgoal[6:] - observation[6:]) <= 0.04)
        else:
            return (np.linalg.norm(subgoal[19:22] - observation[19:22]) <= 0.04 and
                    np.linalg.norm(subgoal[28:31] - observation[28:31]) <= 0.04 and
                    np.linalg.norm(subgoal[37:40] - observation[37:40]) <= 0.04)

def add_noise(subgoal, noise_std, rng):
    noise = jax.random.normal(rng, subgoal.shape) * noise_std
    return subgoal + noise

# rng, cur_rng = jax.random.split(rng)
#             # noise = jax.random.normal(cur_rng, subgoal.shape) * self.config['noise_std']
#             # subgoal = subgoal + noise

#             subgoal = add_noise(subgoal, self.config['noise_std'], rng)

class Sampler(ABC):

    @abstractmethod
    def sample(self, **kwargs):
        """
        Sample a goal based on the current observation.
        
        Args:
            observation: The current observation from the environment.
            **kwargs: Additional arguments that may be needed for sampling.
        
        Returns:
            A sampled goal.
        """
        pass

##=========== SIMPLEST GCFQL SAMPLER ===========##
@struct.dataclass
class SimpleGCFQLSampler(Sampler):
    train_dataset_obs : jnp.ndarray = None
    train_dataset_obs_subset : jnp.ndarray = None
    train_dataset_actions_subset : jnp.ndarray = None
    value_agent : flax.struct.PyTreeNode = None
    goal_ob : jnp.ndarray = None
    goal_ob_subset : jnp.ndarray = None
    temperature: float = 1.0
    beta: float = 1.0
    subgoal_to_goal_values : jnp.ndarray = None

    def sample(self, observation, rng, goal_only=False):
        if goal_only:
            return self.goal_ob, None, None

        return self.sample_helper(observation, rng)

    @jax.jit
    def sample_helper(self, observation, rng):
        num = self.train_dataset_obs_subset.shape[0]
        key, rng = jax.random.split(rng)
        observation_stack = jnp.repeat(observation[None], num, axis=0)
        actions = self.value_agent.sample_actions(observations=observation_stack, goals=self.train_dataset_obs_subset, seed=key)
        curr_to_subgoal_values = self.value_agent.network.select('critic')(observations=observation_stack, goals=self.train_dataset_obs_subset, actions=actions).mean(axis=0)

        m = jnp.mean(curr_to_subgoal_values)
        v = jnp.std(curr_to_subgoal_values)

        subgoal_to_goal_values = jnp.where(curr_to_subgoal_values > m + 1 * v, self.subgoal_to_goal_values, -jnp.inf)
        distribution = jax.nn.softmax(subgoal_to_goal_values / self.temperature)
        chosen_idx = jax.random.choice(key, jnp.arange(num), p=distribution)
        subgoal = self.train_dataset_obs_subset[chosen_idx]

        return subgoal, distribution, (-1, -1)

    # @jax.jit
    # def step(self, observations, goals=None, temperature=None, seed=None):

    def reached_subgoal(self, env_name, subgoal, observation):
        if 'humanoidmaze' in env_name:
            return np.linalg.norm(subgoal[:2] - observation[:2]) <= 0.5
        if 'cube-triple' in env_name:
            if subgoal.shape[0] == 9: # oraclerep
                return (np.linalg.norm(subgoal[:3] - observation[:3]) <= 0.04 and
                        np.linalg.norm(subgoal[3:6] - observation[3:6]) <= 0.04 and
                        np.linalg.norm(subgoal[6:] - observation[6:]) <= 0.04)
            else:
                return (np.linalg.norm(subgoal[19:22] - observation[19:22]) <= 0.04 and
                        np.linalg.norm(subgoal[28:31] - observation[28:31]) <= 0.04 and
                        np.linalg.norm(subgoal[37:40] - observation[37:40]) <= 0.04)

        

    @classmethod
    def create(cls, train_dataset, value_agent, goal_ob, rng, temperature=1.0, beta=1.0, subdataset_obs_size=2000):
        # if 'oraclereps' in train_dataset.dataset:
        #     train_dataset_obs = train_dataset.dataset['oraclereps']
        # if 'oracle_reps' in train_dataset:
        #     train_dataset_obs = train_dataset['oracle_reps']

        #     assert len(train_dataset_obs) > 2000, 'not enough transitions'
        #     # assert goal_ob.shape == train_dataset_obs[0], 'mismatching shape'
        # else:
        if 'oraclereps' in train_dataset:
            train_dataset_obs = train_dataset['oraclereps']
        else:
            train_dataset_obs = train_dataset['observations']
        assert len(train_dataset_obs) > 2000, 'not enough transitions'


        key, rng = jax.random.split(rng)
        sample_idx = jax.random.choice(key, np.arange(train_dataset_obs.shape[0]), shape=(subdataset_obs_size,), replace=False)
        train_dataset_obs_subset = train_dataset_obs[sample_idx]
        train_dataset_actions_subset = train_dataset['actions'][sample_idx]
        goal_ob_subset = jnp.repeat(goal_ob[None], subdataset_obs_size, axis=0)
        # subgoal_to_goal_values = value_agent.network.select('critic')(observations=train_dataset_obs_subset, goals=goal_ob_subset, actions=None).mean(axis=0)
        subgoal_to_goal_values = values_helper(value_agent, train_dataset_obs_subset, goal_ob_subset, train_dataset_actions_subset)

        return cls(train_dataset_obs=train_dataset_obs, train_dataset_obs_subset=train_dataset_obs_subset, value_agent=value_agent, goal_ob=goal_ob, goal_ob_subset=goal_ob_subset, train_dataset_actions_subset=train_dataset_actions_subset, subgoal_to_goal_values=subgoal_to_goal_values)

@jax.jit
def values_helper(value_agent, train_dataset_obs_subset, goal_ob_subset, train_dataset_actions_subset):
    return value_agent.network.select('critic')(observations=train_dataset_obs_subset, goals=goal_ob_subset, actions=train_dataset_actions_subset).mean(axis=0)

##=========== START GCFQL SAMPLER ===========##

@struct.dataclass
class GCFQLSampler(Sampler):
    pairwise_values: jnp.ndarray = None
    freqs: jnp.ndarray = None
    pos: jnp.ndarray = None  # shape (N, 2)
    # discount: float
    goal_idx: int = 0.0
    goal_ob: jnp.ndarray = None
    # radius: float
    temperature: float = 1.0
    beta: float = 1.0

    @jax.jit
    def get_closest_idx(self, pos):
        """
        Get the index of the closest position in the dataset to the given position.
        """
        # norms_squared = jnp.sum((self.pos - pos) ** 2, axis=1)
        norms = jnp.linalg.norm(self.pos - pos, axis=1)
        return jnp.argmin(norms)

    @jax.jit
    def sample_with_agent(self, observation, agent, goal=None, key=None):
        goal_ob = observation.copy()
        if goal is not None:
            goal_ob = goal_ob.at[:2].set(goal[:2])
        else:
            goal_ob = goal_ob.at[:2].set(self.goal_ob[:2])

        goal_ob = jnp.repeat(goal_ob[None], self.pos.shape[0], axis=0)
        potential_subgoals = jnp.tile(observation[2:], (self.pos.shape[0], 1))
        potential_subgoals = jnp.concatenate([self.pos, potential_subgoals], axis=1)

        actions = agent.sample_actions(observations=potential_subgoals, goals=goal_ob, seed=key)
        subgoal_to_goal_values = agent.network.select('critic')(observations=potential_subgoals, goals=goal_ob, actions=actions).mean(axis=0)

        observation = jnp.repeat(observation[None], self.pos.shape[0], axis=0)
        actions = agent.sample_actions(observations=observation, goals=potential_subgoals, seed=key)
        curr_to_subgoal_values = agent.network.select('critic')(observations=observation, goals=potential_subgoals, actions=actions).mean(axis=0)

        m = jnp.mean(curr_to_subgoal_values)
        v = jnp.std(curr_to_subgoal_values)
        subgoal_to_goal_values = jnp.where(curr_to_subgoal_values > m + v, subgoal_to_goal_values, -jnp.inf)
        
        distribution = jax.nn.softmax(subgoal_to_goal_values / self.temperature)
        chosen_idx = jax.random.choice(key, jnp.arange(len(self.pos)), p=distribution)
        subgoal = potential_subgoals[chosen_idx]
        return subgoal, distribution

    @jax.jit
    def sample(self, observation, key):
        curr_idx = self.get_closest_idx(observation[:2])
        subgoal_to_goal_values = self.pairwise_values[:, self.goal_idx]
        # subgoal_to_goal_values = subgoal_to_goal_values + (self.beta / jnp.sqrt(self.freqs[:, self.goal_idx] + 1) - 1)
        subgoal_to_goal_values = subgoal_to_goal_values - 0.05 * jnp.sqrt(self.freqs[:, self.goal_idx])

        # norms = jnp.linalg.norm(self.pos - observation[:2], axis=1)
        # subgoal_to_goal_values = jnp.where(norms < self.radius, subgoal_to_goal_values, -jnp.inf)

        # Don't want to use frequency for the initial filtering, which limits too much.
        curr_to_subgoal_values = self.pairwise_values[curr_idx, :] # + (self.beta / jnp.sqrt(self.freqs[curr_idx, :] + 1) - 1)
        m = jnp.mean(curr_to_subgoal_values)
        std = jnp.std(curr_to_subgoal_values)
        # dynamical_dists = self.dynamical_distance(curr_to_subgoal_values)
        # m = jnp.mean(dynamical_dists)
        # std = jnp.std(dynamical_dists)

        subgoal_to_goal_values = jnp.where(curr_to_subgoal_values > m + 1 * std, subgoal_to_goal_values, -jnp.inf)
        # subgoal_to_goal_values = jnp.where((dynamical_dists < 50), subgoal_to_goal_values, -jnp.inf,)
        
        distribution = jax.nn.softmax(subgoal_to_goal_values / self.temperature)
        chosen_idx = jax.random.choice(key, jnp.arange(len(self.pos)), p=distribution)
        pos = self.pos[chosen_idx]
        subgoal = observation.at[:2].set(pos)
        # if add_info:
            # return subgoal, {'distribution': distribution}

        return subgoal, distribution, (curr_idx, chosen_idx)

    def update_coord(self, curr_idx, chosen_idx, successful):
        curr_value = self.freqs[curr_idx, chosen_idx]
        if successful:
            new_value = curr_value + 1
        freqs = self.freqs.at[curr_idx, chosen_idx].set(new_value)
        
        # return f'successful update to {new_value}'
        return self.replace(pairwise_values=self.pairwise_values, freqs=freqs, pos=self.pos, goal_idx=self.goal_idx, temperature=self.temperature, goal_ob=self.goal_ob), f'successful update to {new_value}'

    def update_coord_freq(self, a_idx=None, b_idx=None): # chosen_idx):
        assert (a_idx is not None) or (b_idx is not None), 'at least one idx must be not None'
        curr_values = self.freqs[a_idx, b_idx]
        new_values = curr_values + 1
        # new_values = curr_values + 0.1
        freqs = self.freqs.at[a_idx, b_idx].set(new_values)
        
        return self.replace(pairwise_values=self.pairwise_values, freqs=freqs, pos=self.pos, goal_idx=self.goal_idx, temperature=self.temperature, goal_ob=self.goal_ob), f'successful update'

    @classmethod
    def create_with_idx(cls, train_dataset, goal_idx, temperature, **kwargs):
        samples_idx = np.random.choice(np.arange(len(train_dataset.dataset['observations'])), size=2000, replace=False)
        observations = train_dataset.dataset['observations'][samples_idx]
        pos = observations[:, :2]

        # goal_idx is a placeholder for now
        return cls(pos=pos, goal_idx=goal_idx, temperature=temperature)
    
    @classmethod
    def create(cls, train_dataset, goal_ob, temperature, beta=1.0, **kwargs):
        samples_idx = np.random.choice(np.arange(len(train_dataset.dataset['observations'])), size=2000, replace=False)
        observations = train_dataset.dataset['observations'][samples_idx]
        pos = observations[:, :2]

        # goal_idx is a placeholder for now
        return cls(pos=pos, goal_ob=goal_ob, temperature=temperature, beta=beta)

    @classmethod
    def precompute_create(cls, env_name, train_dataset, value_agent, goal_ob, rng, temperature=1.0, num_samples=2000, **kwargs):
        key, rng = jax.random.split(rng)

        samples_idx = jax.random.choice(key, np.arange(len(train_dataset.dataset['observations'])), shape=(num_samples,), replace=False)
        if 'oraclereps' in train_dataset.dataset:
            pos = train_dataset.dataset['oraclereps'][samples_idx]
            goal_ob = goal_ob
        else:
            samples = train_dataset.dataset['observations'][samples_idx]

            pos = samples[:, :2]
            ob_end = samples[0][2:]
            goal_ob = jnp.concatenate([jnp.array(goal_ob), ob_end])

        with open('gcfql-values.pkl', 'rb') as f:
            values_map = pickle.load(f)
        if (env_name, num_samples) in values_map:
            print('preloaded')
            pairwise_values, pos = values_map[(env_name, num_samples)]
            goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_ob[:2]), axis=1))
            freqs = jnp.zeros((pos.shape[0], pos.shape[0]))
            return cls(pairwise_values=pairwise_values, freqs=freqs, pos=pos, goal_idx=goal_idx, goal_ob=goal_ob)
            
        print('computing values')

        ob_end_repeated = jnp.tile(ob_end, (num_samples, 1))
        pos_obs = jnp.concatenate([pos, ob_end_repeated], axis=1)

        step_size=100
        pairwise_values = jnp.zeros((num_samples, num_samples))

        for i in tqdm(range(0, num_samples, step_size)):
            pos_obs_i_pre = pos_obs[i:min(i + step_size, num_samples)]
            for j in tqdm(range(0, num_samples, step_size)):
                pos_obs_j_pre = pos_obs[j:min(j + step_size, num_samples)]

                pairwise_values_ij = compute_gcfql_values(pos_obs_i_pre, pos_obs_j_pre, value_agent, key)
                pairwise_values = pairwise_values.at[i:min(i + step_size, num_samples), j:min(j + step_size, num_samples)].set(pairwise_values_ij)

        goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_ob[:2]), axis=1))
        values_map[(env_name, num_samples)] = (pairwise_values, pos)

        with open('gcfql-values.pkl', 'wb') as f:
            pickle.dump(values_map, f)

        freqs = jnp.zeros((pos.shape[0], pos.shape[0]))

        return cls(pairwise_values=pairwise_values, freqs=freqs, pos=pos, goal_idx=goal_idx, temperature=temperature, goal_ob=goal_ob)


@jax.jit
def compute_gcfql_values(pos_obs_i, pos_obs_j, value_agent, key):
    pos_obs_i_ex = jnp.repeat(pos_obs_i, pos_obs_i.shape[0], axis=0)
    pos_obs_j_ex = jnp.tile(pos_obs_j, (pos_obs_j.shape[0], 1))
    actions = value_agent.sample_actions(observations=pos_obs_i_ex, goals=pos_obs_j_ex, seed=key)

    values = value_agent.network.select('critic')(observations=pos_obs_i_ex, goals=pos_obs_j_ex, actions=actions).mean(axis=0)
    pairwise_values_ij = jnp.reshape(values, (pos_obs_i.shape[0], -1))

    return pairwise_values_ij


##=========== END GCFQL SAMPLER===========##

@struct.dataclass
class GCVSampler(Sampler):
    pairwise_values: jnp.ndarray
    pos: jnp.ndarray  # shape (N, 2)
    discount: float
    goal_idx: int
    radius: float
    temperature: float = 1.0

    @jax.jit
    def dynamical_distance(self, v):
        """
        Compute the dynamical distance based on the value v.
        """
        return jnp.log(jnp.clip(v * (1 - self.discount) + 1, 1e-4, 1.0)) / jnp.log(self.discount)

    @jax.jit
    def get_closest_idx(self, pos):
        """
        Get the index of the closest position in the dataset to the given position.
        """
        # norms_squared = jnp.sum((self.pos - pos) ** 2, axis=1)
        norms = jnp.linalg.norm(self.pos - pos, axis=1)
        return jnp.argmin(norms)

    @jax.jit
    def sample(self, observation, key=None, add_info=False):
        curr_idx = self.get_closest_idx(observation[:2])
        subgoal_to_goal_values = self.pairwise_values[:, self.goal_idx]

        norms = jnp.linalg.norm(self.pos - observation[:2], axis=1)
        subgoal_to_goal_values = jnp.where(norms < self.radius, subgoal_to_goal_values, -jnp.inf)

        curr_to_subgoal_values = self.pairwise_values[curr_idx, :]
        dynamical_dists = self.dynamical_distance(curr_to_subgoal_values)
        m = jnp.mean(dynamical_dists)
        std = jnp.std(dynamical_dists)

        subgoal_to_goal_values = jnp.where((dynamical_dists < m - 1 * std), subgoal_to_goal_values, -jnp.inf,)
        # subgoal_to_goal_values = jnp.where((dynamical_dists < 50), subgoal_to_goal_values, -jnp.inf,)
        
        distribution = jax.nn.softmax(subgoal_to_goal_values / self.temperature)
        chosen_idx = jax.random.choice(key, jnp.arange(len(self.pos)), p=distribution)
        pos = self.pos[chosen_idx]
        subgoal = observation.at[:2].set(pos)
        # if add_info:
            # return subgoal, {'distribution': distribution}
        return subgoal, distribution

    @classmethod
    # def from_files(cls, pairwise_values, pairwise_values_labels, discount, goal_xy, **kwargs):
    #     with open(pairwise_values, "rb") as f:
    #         pairwise_values = pickle.load(f)
    #     with open(pairwise_values_labels, 'rb') as f:
    #         pos = pickle.load(f)[:, :2]
    #     pairwise_values = pairwise_values[:2, :2]
    #     pos = pos[:2]
    #     goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_xy), axis=1))
    #     return cls(pairwise_values=pairwise_values, pos=pos, discount=discount, goal_idx=goal_idx)
    def from_files(cls, env_name, seed, ckpts_dir, discount, goal_xy, radius, temperature, **kwargs):
        with open('gciql-values.pkl', 'rb') as f:
            gciql_map = pickle.load(f)
        canonical_env_name = "-".join(env_name.split("-")[:-3]) + "-v0"

        file_name = gciql_map[(canonical_env_name, seed)] + '-values.pkl'
        path = os.path.join(ckpts_dir, file_name)
        with open(path, 'rb') as f:
            pairwise_values = pickle.load(f)

        path = path.replace('-values.pkl', '-pos.pkl')
        with open(path, 'rb') as f:
            pos = pickle.load(f)

        goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_xy), axis=1))
        return cls(pairwise_values=pairwise_values, pos=pos, discount=discount, goal_idx=goal_idx, radius=radius, temperature=temperature)
    

@struct.dataclass
class NaiveSampler(Sampler):
    optimal_path: jnp.ndarray
    subgoal_idx: int = 0

    def sample(self, observation):
        new_subgoal_idx = self.subgoal_idx
        if self.subgoal_idx == 0:
            new_subgoal_idx += 1

        elif self.subgoal_idx == self.optimal_path.shape[0] - 1:
            new_subgoal_idx = self.subgoal_idx

        else:
            xy = observation[:2]
            closest_idx = np.argmin(jnp.linalg.norm(self.optimal_path[:, :2] - xy, axis=1))
            if closest_idx > new_subgoal_idx:
                new_subgoal_idx = closest_idx
            subgoal_xy = self.optimal_path[new_subgoal_idx][:2]
            if jnp.linalg.norm(xy - subgoal_xy) <= 1:
                new_subgoal_idx += 1
            # prev_xy = self.optimal_path[new_subgoal_idx - 1][:2]
            # next_xy = self.optimal_path[new_subgoal_idx + 1][:2]

            # dist_prev = np.linalg.norm(xy - prev_xy)
            # dist_next = np.linalg.norm(xy - next_xy)

            # if dist_prev > dist_next:
            #     new_subgoal_idx += 1
        
        return self.optimal_path[new_subgoal_idx], self.replace(optimal_path=self.optimal_path, subgoal_idx=new_subgoal_idx)
    
    def reset(self):
        return self.replace(optimal_path=self.optimal_path, subgoal_idx=0)
    
    @classmethod
    def create(cls, goal_obs, optimal_path):
        optimal_path_exp = []
        for i, xy in enumerate(optimal_path):
            if i == len(optimal_path) - 1:
                optimal_path_exp.append(xy)
            else:
                optimal_path_exp.append(xy)
                optimal_path_exp.append(0.5 * xy + 0.5 * optimal_path[i + 1])

        optimal_path = np.array(optimal_path_exp)
        goal_obs_rest = goal_obs[2:]

        goal_obs_rest = np.tile(goal_obs_rest, (optimal_path.shape[0], 1))
        optimal_path_obs = np.concatenate([optimal_path, goal_obs_rest], axis=1)


        return cls(optimal_path=optimal_path_obs)

@struct.dataclass
class CFGRLSampler(Sampler):
    pairwise_values: jnp.ndarray
    pos: jnp.ndarray  # shape (N, 2)
    discount: float
    goal_idx: int

    @jax.jit
    def dynamical_distance(self, v):
        """
        Compute the dynamical distance based on the value v.
        """
        return jnp.log(jnp.clip(v * (1 - self.discount) + 1, 1e-4, 1.0)) / jnp.log(self.discount)

    @jax.jit
    def get_closest_idx(self, pos):
        """
        Get the index of the closest position in the dataset to the given position.
        """
        # norms_squared = jnp.sum((self.pos - pos) ** 2, axis=1)
        norms = jnp.linalg.norm(self.pos - pos, axis=1)
        return jnp.argmin(norms)

    @jax.jit
    def sample(self, observation, key=None):
        curr_idx = self.get_closest_idx(observation[:2])
        subgoal_to_goal_values = self.pairwise_values[:, self.goal_idx]

        norms = jnp.linalg.norm(self.pos - observation[:2], axis=1)
        subgoal_to_goal_values = jnp.where(norms < 3, subgoal_to_goal_values, -jnp.inf)

        curr_to_subgoal_values = self.pairwise_values[curr_idx, :]
        dynamical_dists = self.dynamical_distance(curr_to_subgoal_values)
        m = jnp.mean(dynamical_dists)
        std = jnp.std(dynamical_dists)

        subgoal_to_goal_values = jnp.where((dynamical_dists < m - 1 * std), subgoal_to_goal_values, -jnp.inf,)
        # subgoal_to_goal_values = jnp.where((dynamical_dists < 50), subgoal_to_goal_values, -jnp.inf,)
        
        distribution = jax.nn.softmax(subgoal_to_goal_values)
        chosen_idx = jax.random.choice(key, jnp.arange(len(self.pos)), p=distribution)
        pos = self.pos[chosen_idx]
        subgoal = observation.at[:2].set(pos)
        return subgoal

    @classmethod
    def from_files(cls, pairwise_values, pairwise_values_labels, discount, goal_xy, **kwargs):
        with open(pairwise_values, "rb") as f:
            pairwise_values = pickle.load(f)
        with open(pairwise_values_labels, 'rb') as f:
            pos = pickle.load(f)
        goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_xy), axis=1))
        return cls(pairwise_values=pairwise_values, pos=pos, discount=discount, goal_idx=goal_idx)
    
    @classmethod
    def from_agent(cls, value_agent, train_dataset, discount, goal_xy, **kwargs):
        """
        Create a CFGSampler from an agent's value function.
        
        Args:
            value_agent: The agent whose value function is used to create the sampler.
            discount: The discount factor for the value function.
            goal_xy: The coordinates of the goal.
        
        Returns:
            An instance of CFGSampler.
        """
        N = int(train_dataset.size * 0.01)
        # subset = np.random.choice(train_dataset.dataset["observations"], size=N, replace=False)
        subset_idxs = np.random.choice(np.arange(train_dataset.dataset.size), size=N, replace=False)
        subset = train_dataset.dataset['observations'][subset_idxs]
        subset = jnp.array(subset)
        pos = subset[:, :2]

        value_net = value_agent.network.select('value')
        pairwise_values = jnp.zeros((N, N))

        step_size = 100
        for i in tqdm(range(0, N, step_size)):
            row_end = min(i + step_size, N)
            subset_i = subset[i:row_end]
            for j in tqdm(range(0, N, step_size)):
                endp = min(j + step_size, N)
                subset_j = subset[j:endp]
                subset_i_prep = np.repeat(subset_i, subset_j.shape[0], axis=0)
                subset_j_prep = np.tile(subset_j, (subset_i.shape[0], 1))
                flattened_values = value_net(subset_i_prep, subset_j_prep)
                grid_values = jnp.reshape(flattened_values, (subset_i.shape[0], -1))
                pairwise_values = pairwise_values.at[i:row_end, j:endp].set(grid_values)
        pairwise_values = jnp.array(pairwise_values)


        goal_idx = jnp.argmin(jnp.linalg.norm(pos - jnp.array(goal_xy), axis=1))
        return cls(pairwise_values=pairwise_values, pos=pos, discount=discount, goal_idx=goal_idx)