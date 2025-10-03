from functools import partial
from matplotlib.pyplot import step
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import ml_collections
from requests import get 
from utils.datasets import Dataset, ReplayBuffer
import os
import tqdm
from utils.samplers import to_oracle_rep
from utils.statistics import statistics
import gymnasium
from utils.plot_utils import plot_data, calculate_all_cells, bfs
from agents.rnd import RND
from utils.flax_utils import restore_rb

def get_goal(vertex_cells_xy, rnd, temp=1.0):
    rewards, rnd_stats = rnd.get_reward(observations=vertex_cells_xy, stats=True, actions=None)
    # goal_idx = np.argmax(rewards)
    # distribution = jax.nn.softmax(rewards)
    # distribution = jax.nn.softmax(rewards / temp)
    distribution = np.exp(rewards / temp) / np.sum(np.exp(rewards / temp))
    goal_xy = vertex_cells_xy[np.random.choice(len(vertex_cells_xy), p=distribution)]
    return goal_xy, rnd_stats, rewards
    # goal_xy = vertex_cells_xy[goal_idx]

@struct.dataclass
class WithRND:
    '''
    Sample subgoals to navigate to.
    '''
    
    @classmethod
    def create(cls, original_dataset, config, train_dataset, env, seed, save_dir, start_ij, wandb, agent, **kwargs):
        '''
        Should return an expanded dataset that combines the original dataset with new datasets.
        '''
        # if wandb.run.summary.get('_datacollection_checkpoint', 0) > 0:
        #     dataset = restore_rb(save_dir, int(wandb.run.summary['_datacollection_checkpoint']))
        #     rbsize = train_dataset.size + config['collection_steps']
        #     replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(dataset), rbsize)
        #     replay_buffer.pointer = wandb.run.summary['_datacollection_checkpoint.size']
        #     replay_buffer.size = wandb.run.summary['_datacollection_checkpoint.size']
        # else:
        # rbsize = original_dataset.size + config['collection_steps']
        # replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(original_dataset.dataset), rbsize)

        # rng = jax.random.PRNGKey(seed)
        env_name = env.spec.id
        # infos = env.task_infos

        canonical_env_name = env_name
        stats = statistics[canonical_env_name](env=env)

        # env = gymnasium.make(
        #     env.spec.id,
        #     terminate_at_goal=False,
        #     use_oracle_rep=True,
        #     max_episode_steps=config['max_episode_steps'],
        # )

        ##=========== SET UP GOAL SAMPLING ===========##

        vertex_cells_xy = to_oracle_rep(train_dataset.dataset['observations'], env=env)

        rnd_config = ml_collections.ConfigDict(
            dict(
                agent_name='rnd',
                lr=config['rnd_lr'],
                coeff=config['rnd_coeff'],
                hidden_dims=config['rnd_hidden_dims'],
            )
        )

        observation_example = vertex_cells_xy[0:1][0]
        action_example = None
        rnd = RND.create(config['rnd_seed'], observation_example=observation_example, action_example=action_example, config=rnd_config)

        ##=========== SET UP PLOTTING CODE ===========##
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

        # goal_ij = vertex_cells[np.random.choice(len(vertex_cells))]
        # goal_xy = env.unwrapped.ij_to_xy(goal_ij)

        # rewards, rnd_stats = rnd.get_reward(observations=vertex_cells_xy, stats=True, actions=None)
        # goal_idx = np.argmax(rewards)
        # # goal_ij = vertex_cells[goal_idx]
        # goal_xy = vertex_cells_xy[goal_idx]
        goal_xy, rnd_stats, _ = get_goal(vertex_cells_xy, rnd)
        goal_ij = env.unwrapped.xy_to_ij(goal_xy)

        ##=========== SET UP PLOTTING CODE ===========##

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=goal_ij)))
        env.unwrapped.set_goal(goal_ij)

        data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal_xy[0])
        data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal_xy[1])
        data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], 0)

        collection_info = {
            'num_successes': 0,
            'num_goals': 1,
        }


        return cls()
        # start_i = 1 if not PREEMPTED else int(wandb.run.summary['_datacollection_checkpoint']) + 1
        # if wandb.run.summary.get('_datacollection_checkpoint', 0) > 0:
        #     start_i = int(wandb.run.summary['_datacollection_checkpoint']) + 1
        # else:
        #     start_i = 1

        
        ##=========== START DATA COLLECTION ===========##

        # log_start_i = wandb.run.step
    def collect_step(self, replay_buffer, env, rng):
        # for collect_i in tqdm.tqdm(range(start_i, config['collection_steps'] + 1)):

            curr_rng, rng = jax.random.split(rng)
            action = agent.sample_actions(
                observations=ob,
                goals=goal_xy,
                seed=curr_rng,
            )

            action = action + np.random.normal(0, config['noise'], size=action.shape)
            action = np.clip(action, -1, 1)
            next_ob, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            # success = info['success']
            if np.linalg.norm(ob[:2] - goal_xy) < 0.5:
                success = True
            else:
                success = False

            if success: # TODO: update if not navigate

                goal_xy, rnd_stats, _ = get_goal(vertex_cells_xy, rnd)
                goal_ij = env.unwrapped.xy_to_ij(goal_xy)
                env.unwrapped.set_goal(goal_ij)

                data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal_xy[0])
                data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal_xy[1])
                collection_info['num_goals'] += 1
                data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], collection_info['num_goals'])

                collection_info['num_successes'] += 1

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

            rnd, rnd_info = rnd.update(batch={
                'observations': tran['oracle_reps'],
                'actions': None,
            })

            for k, v in rnd_info.items():
                import ipdb; ipdb.set_trace()
                wandb.log({f"data_collection/rnd_{k}": v, 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)

            data_to_plot['replay buffer']['x'] = np.append(data_to_plot['replay buffer']['x'], ob[0])
            data_to_plot['replay buffer']['y'] = np.append(data_to_plot['replay buffer']['y'], ob[1])
            data_to_plot['replay buffer']['c'] = np.append(data_to_plot['replay buffer']['c'], collect_i)

            stats.log_episode(ob, action)

            if done:
                done = False
                # goal = agent.propose_goals(observations=ob[None], goals=np.array([20,20]), rng=rng)
                # goal_ij = env.unwrapped.xy_to_ij(goal_xy)

                # rewards, rnd_stats = rnd.get_reward(observations=vertex_cells_xy, stats=True, actions=None)
                # goal_idx = np.argmax(rewards)
                # # goal_ij = vertex_cells[goal_idx]
                # goal_xy = vertex_cells_xy[goal_idx]
                goal_xy, rnd_stats, rewards = get_goal(vertex_cells_xy, rnd)
                goal_ij = env.unwrapped.xy_to_ij(goal_xy)
                env.unwrapped.set_goal(goal_ij)

                for k, v in rnd_stats.items():
                    import ipdb; ipdb.set_trace()
                    wandb.log({f"data_collection/rnd_{k}": v, 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)

                data_to_plot['goals']['x'] = np.append(data_to_plot['goals']['x'], goal_xy[0])
                data_to_plot['goals']['y'] = np.append(data_to_plot['goals']['y'], goal_xy[1])
                collection_info['num_goals'] += 1
                data_to_plot['goals']['c'] = np.append(data_to_plot['goals']['c'], collection_info['num_goals'])

                fig_name = plot_data(
                    {'rnd_reward': {
                        'x': vertex_cells_xy[::10, 0],
                        'y': vertex_cells_xy[::10, 1],
                        'c': rewards[::10],
                        's': 20,
                        'cmap': 'plasma',
                    }},
                    save_dir=save_dir
                )

                import ipdb; ipdb.set_trace()
                wandb.log({f"data_collection/rnd_reward_viz": v, 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)
                print(f"Plotted RND rewards to {fig_name}")
                os.remove(fig_name)

                ob, _ = env.reset(options=dict(task_info=dict(init_ij=start_ij, goal_ij=goal_ij))) # TODO: use goal_ij placeholder
                # ob, _ = env.reset()
            else:
                ob = next_ob

            if collect_i % config['plot_interval'] == 0:
                for k, v in stats.get_statistics().items():
                    
                    import ipdb; ipdb.set_trace()
                    wandb.log({f"data_collection/{k}": v, 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)

                fig_name = plot_data(
                    data_to_plot,
                    save_dir=save_dir
                )

                wandb.log({"data_collection/replay_buffer_viz": wandb.Image(fig_name), 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)
                print(f"Plotted data to {fig_name}")
                os.remove(fig_name)

                for k, v in info.items():
                    import ipdb; ipdb.set_trace()
                    wandb.log({f"data_collection/{k}": v, 'data_collection/collect_i': collect_i}, step=log_start_i + collect_i)

            if collect_i != 0 and collect_i % config['save_data_interval'] == 0:
                print(f"Collected {collect_i} steps")
                np.savez(os.path.join(save_dir, f"data-{collect_i}.npz"), **replay_buffer)
                print(f"Saved dataset to {os.path.join(save_dir, f'data-{collect_i}.npz')}")
                wandb.run.summary['_datacollection_checkpoint'] = collect_i
                wandb.run.summary['_datacollection_checkpoint.size'] = replay_buffer.size
                # wandb.run.summary['_datacollection_checkpoint.pointer'] = replay_buffer.pointer

        # np.savez(os.path.join(save_dir, f"data-{collect_i}.npz"), **replay_buffer)
        # wandb.run.summary['_datacollection_checkpoint'] = collect_i
        # wandb.run.summary['_datacollection_checkpoint.size'] = replay_buffer.size
        # wandb.run.summary['_datacollection_checkpoint.pointer'] = replay_buffer.pointer
        return replay_buffer

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='withrnd',
            collection_steps=1000000,
            save_data_interval=500000,
            plot_interval=10000,
            max_episode_steps=2000, # must be a divisor of collection_steps
            noise=0.0,
            rnd_lr=1e-4,
            rnd_coeff=1.0,
            rnd_hidden_dims=[512, 512, 512],
            rnd_seed=0,
        )
    )
    return config