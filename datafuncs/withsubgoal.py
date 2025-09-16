from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import ml_collections 
from utils.datasets import Dataset, ReplayBuffer
import os
import tqdm
from utils.samplers import to_oracle_rep
from utils.statistics import statistics
import gymnasium
from utils.plot_utils import plot_data, calculate_all_cells, bfs
from utils.restore import restore_rb

@struct.dataclass
class WithSubgoal:
    '''
    Sample subgoals to navigate to.
    '''
    
    def create(original_dataset, config, agent_config, env, seed, save_dir, start_ij, wandb, agent, **kwargs):
        '''
        Should return an expanded dataset that combines the original dataset with new datasets.
        '''
        if wandb.run.summary.get('_datacollection_checkpoint', 0) > 0:
            dataset = restore_rb(save_dir, int(wandb.run.summary['_datacollection_checkpoint']))
            rbsize = original_dataset.size + config['collection_steps']
            replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(dataset), rbsize)
            replay_buffer.pointer = wandb.run.summary['_datacollection_checkpoint.size']
            replay_buffer.size = wandb.run.summary['_datacollection_checkpoint.size']
        else:
            rbsize = original_dataset.size + config['collection_steps']
            replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(original_dataset.dataset), rbsize)

        rng = jax.random.PRNGKey(seed)
        env_name = env.spec.id
        # infos = env.task_infos

        canonical_env_name = env_name
        stats = statistics[canonical_env_name](env=env)

        env = gymnasium.make(
            env.spec.id,
            terminate_at_goal=False,
            use_oracle_rep=True,
            max_episode_steps=config['max_episode_steps'],
        )

        data_to_plot = {}
        all_cells = calculate_all_cells(env)
        all_cells = np.array([env.unwrapped.ij_to_xy(ij) for ij in all_cells])
        data_to_plot['all_cells'] = {
            'x': all_cells[:, 0],
            'y': all_cells[:, 1],
            'alpha': 0.1,
            's': 5,
            'c': 'gray',
        }
        
        data_to_plot['replay buffer'] = {
            'x': np.array([]),
            'y': np.array([]),
            'c': np.array([]),
            's': 10,
            'cmap': 'viridis',
        }
        data_to_plot['goals'] = {
            'x': np.array([]),
            'y': np.array([]),
            's': 15,
            'marker': '*',
            'c': np.array([]),
            'cmap': 'plasma',
        }
        data_to_plot['start'] = {
            'x': [env.unwrapped.ij_to_xy(start_ij)[0]],
            'y': [env.unwrapped.ij_to_xy(start_ij)[1]],
            's': 50,
            'c': 'red',
        }

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=(0,0)))) # TODO: use goal_ij placeholder
        # ob, _ = env.reset()
        # num_random_actions = 40 # if loco_env_type == 'humanoid' else 5
        # for _ in range(num_random_actions):
        #     env.step(env.action_space.sample())
            
        goal = agent.propose_goals(observations=ob[None], goals=np.array([20,20]), rng=rng)

        goal_ij = env.unwrapped.xy_to_ij(goal[0])
        ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=goal_ij)))
        env.unwrapped.set_goal(goal_ij)

        data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal[0][0])
        data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal[0][1])
        data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], 0)

        collection_info = {
            'num_successes': 0,
            'num_goals': 1,
        }

        if wandb.run.summary.get('_datacollection_checkpoint', 0) > 0:
            start_i = int(wandb.run.summary['_datacollection_checkpoint']) + 1
        else:
            start_i = 1
            
        for i in tqdm.tqdm(range(1, config['collection_steps'] + 1)):

            curr_rng, rng = jax.random.split(rng)
            action = agent.sample_actions(
                observations=ob,
                goals=goal[0],
                seed=curr_rng,
            )

            action = action + np.random.normal(0, config['noise'], size=action.shape)
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            success = info['success']

            if success: # TODO: update if not navigate
                # assert 'navigate' in env_name, "success should only be true for navigate tasks"
                goal = agent.propose_goals(observations=ob[None], goals=np.array([20,20]), rng=rng)
                goal_ij = env.unwrapped.xy_to_ij(goal[0])
                env.unwrapped.set_goal(goal_ij)
                data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal[0][0])
                data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal[0][1])
                collection_info['num_goals'] += 1
                data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], collection_info['num_goals'])

                collection_info['num_successes'] += 1

            # if 'antmaze' in env_name and (
            #     'diverse' in env_name or 'play' in env_name or 'umaze' in env_name
            # ):
            #     # Adjust reward for D4RL antmaze.
            #     reward = reward - 1.0

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
            data_to_plot['replay buffer']['x'] = np.append(data_to_plot['replay buffer']['x'], ob[0])
            data_to_plot['replay buffer']['y'] = np.append(data_to_plot['replay buffer']['y'], ob[1])
            data_to_plot['replay buffer']['c'] = np.append(data_to_plot['replay buffer']['c'], i)

            stats.log_episode(ob, action)

            if done:
                done = False
                goal = agent.propose_goals(observations=ob[None], goals=np.array([20,20]), rng=rng)
                goal_ij = env.unwrapped.xy_to_ij(goal[0])
                env.unwrapped.set_goal(goal_ij)
                data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal[0][0])
                data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal[0][1])
                collection_info['num_goals'] += 1
                data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], collection_info['num_goals'])
                ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=goal_ij))) # TODO: use goal_ij placeholder
                # ob, _ = env.reset()
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

                for k, v in info.items():
                    wandb.log({f"data_collection/{k}": v}, step=i)

            if i != 0 and i % config['save_data_interval'] == 0:
                print(f"Collected {i} steps")
                np.savez(os.path.join(save_dir, f"data-{i}.npz"), **replay_buffer)
                wandb.run.summary['_datacollection_checkpoint'] = i
                wandb.run.summary['_datacollection_checkpoint.size'] = replay_buffer.size
                print(f"Saved dataset to {os.path.join(save_dir, f'data-{i}.npz')}")

        return replay_buffer

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='withsubgoal',
            collection_steps=1000000,
            save_data_interval=100000,
            plot_interval=10000,
            max_episode_steps=2000, # must be a divisor of collection_steps
            noise=0.0
        )
    )
    return config