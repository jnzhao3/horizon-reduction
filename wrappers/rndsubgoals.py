import ml_collections
import jax
from typing import Any
from flax import struct
from agents.rnd import RND
from utils.datasets import GCDataset
import jax.numpy as jnp
import numpy as np
import tqdm

@struct.dataclass
class RNDSubgoals:
    '''
    Sample subgoals, then use subgoal generator to navigat to.
    '''

    agent: Any
    rnd: Any
    config: ml_collections.ConfigDict
    potential_goals: Any = None
    curr_goal: Any = None

    @classmethod
    def create(cls, agent, train_dataset, config: ml_collections.ConfigDict, **kwargs):
        rnd_config = ml_collections.ConfigDict(
            dict(
                agent_name='rnd',
                lr=config['rnd_lr'],
                coeff=config['rnd_coeff'],
                hidden_dims=config['rnd_hidden_dims'],
            )
        )

        if type(train_dataset) == GCDataset:
            train_dataset = train_dataset.dataset

        observation_example = train_dataset['oracle_reps'][0]
        action_example = train_dataset['actions'][0]
        assert 'oracle_reps' in train_dataset, 'RND needs oracle_reps in the dataset'
        potential_goals = train_dataset['oracle_reps']
        
        rnd = RND.create(config['rnd_seed'], observation_example=observation_example, action_example=action_example, config=rnd_config)

        if config.get('pre_init', False):
            print('Pre-initializing RND with dataset...')

            for i in tqdm.tqdm(range(0, train_dataset['observations'].shape[0], 1)):
                rnd, rnd_info = rnd.update(
                    batch={
                        'observations': train_dataset['oracle_reps'][i:i + 1, :],
                        'actions': None,
                    }
                )

                if i % 10000 == 0:
                    # print(f'Pre-initializing RND: {i}/{train_dataset['ob'].shape[0]}')
                    print(f'RND Info at step {i}:', rnd_info)
            print('Done pre-initializing RND.')

        return cls(agent=agent, rnd=rnd, potential_goals=potential_goals, config=config)
    
    def pre(self, **kwargs):
        if self.curr_goal is None:
            curr_goal, rnd_stats = self.get_goal(observations=kwargs['observations'], rng=kwargs['rng'])
            # rnd_stats.update({'goals': curr_goal})
            # return self, rnd_stats
            return self.replace(curr_goal=curr_goal), rnd_stats
        # else:
        return self, {}

    def get_goal(self, observations, rng):
        subset = np.random.choice(self.potential_goals.shape[0], size=min(1000, self.potential_goals.shape[0]), replace=False)
        subset_goals = self.potential_goals[subset]
        rewards, rnd_stats = self.rnd.get_reward(observations=subset_goals, actions=None, stats=True)
        # TODO: make this random sample then select, to add noise
        goal_xy = self.potential_goals[jnp.argmax(rewards)]

        subgoal = self.agent.propose_goals(observations=observations[None], goals=goal_xy[None], rng=rng)
        return subgoal[0], rnd_stats
    
    def sample_actions(self, observations, goals=None, seed=None, pre_info=None, **kwargs):
        action_info = {}
        # goals = pre_info['curr_goal']
        goals = self.curr_goal
        actions = self.agent.sample_actions(observations=observations, goals=goals, seed=seed)
        return actions, action_info
    
    def post(self, transition, **kwargs):
        rnd, rnd_info = self.rnd.update(batch={
            'observations': transition['oracle_reps'],
            'actions': None,
        })
        post_info = rnd_info

        if transition['terminals'] == 1.0:
            curr_goal, rnd_stats = self.get_goal(observations=transition['observations'], rng=kwargs['rng'])
            # rnd_stats.update({'goals': curr_goal})
            # return self, rnd_stats
            # return self.replace(curr_goal=curr_goal), rnd_stats
            post_info.update(rnd_stats)
            # curr_goal = curr_goal + np.random.normal(0, 10, size=curr_goal.shape)  # add some noise
            return self.replace(rnd=rnd, curr_goal=curr_goal), post_info
        return self.replace(rnd=rnd), post_info

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='rndsubgoals',
            rnd_lr=1e-4,
            rnd_coeff=1.0,
            rnd_hidden_dims=[512, 512, 512],
            rnd_seed=0,
            max_episode_steps=2000,  # max episode steps for env
            pre_init=False,
        )
    )
    return config