from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import ml_collections 
from typing import Any
from ogbench.relabel_utils import add_oracle_reps
import wandb
from utils.datasets import Dataset, ReplayBuffer
from horizon_reduction.wrappers.datafuncs_utils import clip_dataset
from ogbench.relabel_utils import add_oracle_reps
from ogbench import load_dataset
import os
from utils.datasets import GCDataset, HGCDataset
import tqdm
from utils.samplers import to_oracle_rep
from utils.statistics import statistics
import gymnasium
from utils.plot_utils import plot_data, calculate_all_cells, bfs

@struct.dataclass
class RandomSteps:
    '''
    Given an original dataset, and specifications for new datasets to combine with,
    create a new dataset that combines them.
    '''
    
    def create(original_dataset, config, agent_config, env, seed, save_dir, start_ij, wandb, **kwargs):
        '''
        Should return an expanded dataset that combines the original dataset with new datasets.
        '''

        # original_dataset_dict = original_dataset.dataset.unfreeze()
        # rbsize = original_dataset_dict['observations'].shape[0] + sum(config['train_data_sizes'])
        rbsize = original_dataset.size + config['collection_steps']
        replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(original_dataset.dataset), rbsize)
        rng = jax.random.PRNGKey(seed)
        env_name = env.spec.id

        # canonical_env_name = '-'.join(env.spec.id.split('-')[:-1]) + '-v0'
        canonical_env_name = env_name
        stats = statistics[canonical_env_name](env=env)

        env = gymnasium.make(
            # FLAGS.env_name,
            env.spec.id,
            terminate_at_goal=False,
            use_oracle_rep=True,
            max_episode_steps=config['max_episode_steps'],
        )

        data_to_plot = []
        all_cells = calculate_all_cells(env)
        all_cells = np.array([env.unwrapped.ij_to_xy(ij) for ij in all_cells])
        data_to_plot.append({
            'x': all_cells[:, 0],
            'y': all_cells[:, 1],
            'alpha': 0.1,
            's': 5,
            'c': 'gray',
            'label': 'all cells',
        })
        data_to_plot.append({
            'x': [env.unwrapped.ij_to_xy(start_ij)[0]],
            'y': [env.unwrapped.ij_to_xy(start_ij)[1]],
            's': 50,
            'c': 'red',
            'label': 'start',
        })
        data_to_plot.append({
            'x': np.array([]),
            'y': np.array([]),
            'c': np.array([]),
            's': 10,
            'label': 'replay buffer',
            'cmap': 'viridis',
        })

        # ob, info = env.reset()
        ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=(0, 0)))) # TODO: use goal_ij placeholder

        for i in tqdm.tqdm(range(1, config['collection_steps'] + 1)):

            curr_rng, rng = jax.random.split(rng)
            action = env.action_space.sample()
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            if 'antmaze' in env_name and (
                'diverse' in env_name or 'play' in env_name or 'umaze' in env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            tran = dict(
                observations=ob,
                actions=action,
                # rewards=reward,
                terminals=float(done),
                # masks=1.0 - terminated,
                next_observations=next_ob,
                qpos=info["qpos"],
                qvel=info["qvel"],
            )

            tran['oracle_reps'] = to_oracle_rep(obs=ob[None], env=env)[0]
            replay_buffer.add_transition(tran)
            data_to_plot[-1]['x'] = np.append(data_to_plot[-1]['x'], ob[0])
            data_to_plot[-1]['y'] = np.append(data_to_plot[-1]['y'], ob[1])
            data_to_plot[-1]['c'] = np.append(data_to_plot[-1]['c'], i)

            stats.log_episode(ob, action)

            if done:
                done = False
                ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=(0, 0)))) # TODO: use goal_ij placeholder
            else:
                ob = next_ob

            if i % config['plot_interval'] == 0:
                for k, v in stats.get_statistics().items():
                    wandb.log({f"data_collection/{k}": v}, step=i)

                fig_name = plot_data(
                    data_to_plot,
                    save_dir=save_dir
                )

                wandb.log({"data_collection/replay_buffer_viz": wandb.Image(fig_name)}, step=i)
                print(f"Plotted data to {fig_name}")
                os.remove(fig_name)

            if i != 0 and i % config['save_data_interval'] == 0:
                print(f"Collected {i} steps")
                np.savez(os.path.join(save_dir, f"data-{i}.npz"), **replay_buffer)
                print(f"Saved dataset to {os.path.join(save_dir, f'data-{i}.npz')}")

        return replay_buffer

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='randomsteps',
            collection_steps=1000000,
            save_data_interval=100000,
            plot_interval=100000,
            max_episode_steps=2000 # must be a divisor of collection_steps
        )
    )
    return config