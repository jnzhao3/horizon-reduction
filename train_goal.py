import glob
import json
from operator import ne
import os
import random
import time
from collections import defaultdict
import sys

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
# from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, GCDataset, HGCDataset
# from utils.evaluation import evaluate_gcfql
from utils.flax_utils import ModuleDict, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb, get_animal
from utils.networks import GCValue, ActorVectorField
from utils.flax_utils import TrainState
import optax
import jax.numpy as jnp
from functools import partial
# from utils.samplers import to_oracle_rep
from ogbench.relabel_utils import add_oracle_reps
import matplotlib.pyplot as plt
import numpy as np
import ogbench

from utils.datasets import Dataset


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval.')
flags.DEFINE_integer('num_datasets', None, 'Number of datasets to use.')
flags.DEFINE_integer('train_data_size', None, 'Size of training data to use (None for full dataset).')

flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 5000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 100000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 500000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_string('json_path', None, 'Path to JSON file with additional parameters.')

flags.DEFINE_integer('eval_episodes', 15, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_bool('use_wandb', True, 'Use Weights & Biases for logging.')
flags.DEFINE_bool('wandb_alerts', True, 'Enable Weights & Biases alerts.')
# flags.DEFINE_string('q_pred_calc', 'sample', 'Method for calculating Q predictions (sample or mean).') # batch

config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)

from train_value import make_env_and_datasets

@partial(jax.jit, static_argnames=('goal_proposer_type'))
def train_step(state, batch, rng, goal_proposer_type):

    if goal_proposer_type == 'default':
        def loss_fn(params, rng_in):
            assert 'low_actor_goals' in batch, "Batch must contain 'low_actor_goals' for goal proposer training."
            batch_size, goal_dim = batch['low_actor_goals'].shape
            rng, x_rng, t_rng = jax.random.split(rng_in, 3)

            x_0 = jax.random.normaal(x_rng, (batch_size, goal_dim))
            x_1 = batch['low_actor_goals']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = state.select('goal_proposer')(
                observations=batch['observations'],
                actions=x_t,
                times=t,
                params=params
            )

            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            info = {
                'bc_flow_loss': bc_flow_loss,
            }
            return bc_flow_loss, info
        
    elif goal_proposer_type == 'actor-gc':
        def loss_fn(params, rng_in):
            assert 'low_actor_goals' in batch, "Batch must contain 'low_actor_goals' for goal proposer training."
            batch_size, goal_dim = batch['low_actor_goals'].shape
            rng, x_rng, t_rng = jax.random.split(rng_in, 3)

            x_0 = jax.random.normaal(x_rng, (batch_size, goal_dim))
            x_1 = batch['low_actor_goals']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = state.select('goal_proposer')(
                observations=batch['observations'],
                goals=batch['actor_goals'],
                actions=x_t,
                times=t,
                params=params
            )

            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            info = {
                'bc_flow_loss': bc_flow_loss,
            }
            return bc_flow_loss, info
    else:
        raise NotImplementedError(f"Unknown goal proposer type: {goal_proposer_type}")
        
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, rng)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(params=new_params, opt_state=new_opt_state)
    return new_state, info

@partial(jax.jit, static_argnames=('goal_proposer_type'))
def val_step(state, batch, rng, goal_proposer_type):

    if goal_proposer_type == 'default':
        def loss_fn(params, rng_in):
            assert 'low_actor_goals' in batch, "Batch must contain 'low_actor_goals' for goal proposer training."
            batch_size, goal_dim = batch['low_actor_goals'].shape
            rng, x_rng, t_rng = jax.random.split(rng_in, 3)

            x_0 = jax.random.normaal(x_rng, (batch_size, goal_dim))
            x_1 = batch['low_actor_goals']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = state.select('goal_proposer')(
                observations=batch['observations'],
                actions=x_t,
                times=t,
                params=params
            )

            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            info = {
                'bc_flow_loss': bc_flow_loss,
            }
            return bc_flow_loss, info
        
    elif goal_proposer_type == 'actor-gc':
        def loss_fn(params, rng_in):
            assert 'low_actor_goals' in batch, "Batch must contain 'low_actor_goals' for goal proposer training."
            batch_size, goal_dim = batch['low_actor_goals'].shape
            rng, x_rng, t_rng = jax.random.split(rng_in, 3)

            x_0 = jax.random.normaal(x_rng, (batch_size, goal_dim))
            x_1 = batch['low_actor_goals']
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = state.select('goal_proposer')(
                observations=batch['observations'],
                goals=batch['actor_goals'],
                actions=x_t,
                times=t,
                params=params
            )

            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            info = {
                'bc_flow_loss': bc_flow_loss,
            }
            return bc_flow_loss, info
    else:
        raise NotImplementedError(f"Unknown goal proposer type: {goal_proposer_type}")
    
    (loss, info) = loss_fn(state.params, rng)
    return loss, info

@jax.jit
def propose_goals(network, observations, goals, rng):
    goal_dim = goals.shape[-1]
    x = jax.random.normal(rng, (observations.shape[0], goal_dim))
    for i in range(network.config['flow_steps']):
        t = jnp.full((*observations.shape[:-1], 1), i / network.config['flow_steps'])
        if network.config['goal_proposer_type'] == 'default':
            vels = network.network.select('goal_proposer')(observations, actions=x, times=t)
            # still need to pass in the goals to make the function happy
        else:
            vels = network.network.select('goal_proposer')(observations, actions=x, goals=goals, times=t)
        x = x + vels / network.config['flow_steps']

@partial(jax.jit, static_argnames=('goal_proposer_type'))
def eval_step(state, observations, rng, goal_proposer_type, actor_goals=None):
    batch = {'observations': observations, 'actor_goals': actor_goals}

    # TODO: implement this
    
    # (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, rng)
    (loss, info) = loss_fn(state.params, rng)
    return loss, info

def main(_):
    # Set up logger.
    with open(f'{FLAGS.restore_path}/flags.json', 'r') as f:
        saved_flags = json.load(f)

    assert 'train_data_size' in saved_flags, "train_data_size not found in saved flags."
    FLAGS.train_data_size = saved_flags['train_data_size']
    print(f"Using train_data_size = {FLAGS.train_data_size}", file=sys.stderr)

    exp_name, info = get_exp_name(FLAGS.seed, config=FLAGS)
    if FLAGS.use_wandb:
        setup_wandb(project='horizon-reduction', group=FLAGS.run_group, name=exp_name)
    else:
        project_name = 'horizon_reduction'
        run_group = 'no_wandb'

    ##=========== LOG MESSAGES TO ERR AND SLACK ===========##
    # start_time = time.time()
    animal = get_animal()
    print(f"\n{animal}\n", exp_name)
    print("\n\n", info, file=sys.stderr)
    print("\n\npython", " ".join(sys.argv), "\n", file=sys.stderr)
    if FLAGS.wandb_alerts:
        wandb.run.alert(title=f"{animal} train_fql run started!", text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}")

    if FLAGS.use_wandb:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    else:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, project_name, run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and datasets.
    config = FLAGS.agent
    if 'humanoidmaze' in FLAGS.env_name:
        assert config['discount'] == 0.995, "Humanoid maze tasks require discount factor of 0.995."

    if FLAGS.dataset_dir is None:
        # datasets = [None]
        raise ValueError("Must provide dataset directory.")
    else:
        # Dataset directory.
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    if FLAGS.num_datasets is not None:
        datasets = datasets[: FLAGS.num_datasets]
    dataset_idx = 0
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx])
    # time_elapsed = time.time() - start_time
    # print(f'Environment and datasets ready after {time_elapsed:.2f} seconds.', file=sys.stderr)

    ##=========== PREPARE DATASETS ===========##
    # start_time = time.time()
    N = int(FLAGS.train_data_size)
    if N > 0:
        new_train_dataset = {}
        if 'valids' in train_dataset:
            idxs = np.where(train_dataset['valids'] == 1)[0]
            idxs = idxs[N]
        else:
            idxs = N
        for k, v in train_dataset.items():
            # Ensure we have a writable host array
            if isinstance(v, np.ndarray):
                arr = v[:idxs].copy()                       # writable copy
            else:
                try:
                    # JAX DeviceArray, memmap, etc. -> force to NumPy writable
                    arr = np.array(v[:idxs], copy=True)
                except Exception:
                    # As a fallback (e.g., PyTorch tensor)
                    try:
                        arr = v[:idxs].clone().cpu().numpy()
                    except Exception:
                        arr = np.array(v[:idxs], copy=True)

            if k == "terminals":
                arr[idxs - 1] = 1  # cast to dtype automatically (bool->True, uint8->1)
            elif k == "valids":
                arr[idxs - 1] = 0

            new_train_dataset[k] = arr

        train_dataset = new_train_dataset

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }
    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    # time_elapsed = time.time() - start_time
    # print(f'Datasets ready after {time_elapsed:.2f} seconds.', file=sys.stderr)

    # start_time = time.time()
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    print(agent.config, file=sys.stderr)

    # Restore agent.
    assert FLAGS.restore_path is not None and FLAGS.restore_epoch is not None, "Must provide restore path and epoch."
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    # time_elapsed = time.time() - start_time
    # print(f'Agent ready after {time_elapsed:.2f} seconds.', file=sys.stderr)

    ##=========== CREATE NEW VALUE FUNCTION NETWORK ===========##
    # start_time = time.time()
    if 'oracle_reps' in example_batch:
        goal_dim = example_batch['oracle_reps'].shape[-1]
        ex_goals = example_batch['oracle_reps']
    else:
        goal_dim = example_batch['observations'].shape[-1]
        ex_goals = example_batch['observations']
    goal_proposer_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=goal_dim, # TODO: double check that
        layer_norm=config['layer_norm'],
    )
    ex_observations = example_batch['observations']
    ex_actions = example_batch['actions']
    ex_times = ex_actions[..., :1]

    network_info = dict(
        goal_proposer = (goal_proposer_def, (ex_observations, ex_goals, ex_goals, ex_times))
    )
    networks = {k: v[0] for k,v in network_info.items()}
    network_args = {k: v[1] for k,v in network_info.items()}
    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config['lr'])
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_rng = jax.random.split(rng)
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # start_time = time.time()
        batch = train_dataset.sample(config['batch_size'])
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        new_rng, step_rng = jax.random.split(rng)
        network, update_info = train_step(
            state=network, batch=batch, rng=step_rng, goal_proposer_type=config['goal_proposer_type']
        )
            
        # time_elapsed = time.time() - start_time
        # print(f'Iteration {i} done after {time_elapsed:.2f} seconds.', file=sys.stderr)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            # start_time = time.time()
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            val_batch = val_dataset.sample(config['batch_size'])
            val_batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), val_batch)

            _, val_info = val_step(
                network, agent, val_batch, new_rng, actions_mode, config['critic_loss_type']
            )
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

            # time_elapsed = time.time() - start_time
            # print(f'Logging {i} done after {time_elapsed:.2f} seconds.', file=sys.stderr)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):

            # start_time = time.time()
            eval_observation, _ = env.reset()
            from utils.plot_utils import bfs, plot_points, plot_replay, calculate_all_cells
            all_cells = calculate_all_cells(env)
            all_cells = [env.unwrapped.ij_to_xy(cell) for cell in all_cells]
            all_cells = np.array(all_cells)


        #     # renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = len(task_infos)

            # time_elapsed = time.time() - start_time
            # print(f'Evaluation setup done after {time_elapsed:.2f} seconds.', file=sys.stderr)

            for task_id in tqdm.trange(1, num_tasks + 1):
                # start_time = time.time()
                goal_ob = env.unwrapped.task_infos[task_id - 1]['goal_ij']
                goal_ob = env.unwrapped.ij_to_xy(goal_ob)
                goal_ob = np.array(goal_ob)
                goal_xy = goal_ob[:2]

                ob, _ = env.reset()
                low_x_bound = all_cells[:,0].min()
                high_x_bound = all_cells[:,0].max()
                low_y_bound = all_cells[:,1].min()
                high_y_bound = all_cells[:,1].max()

                x_to_plot = np.linspace(low_x_bound, high_x_bound / 1.02, 200)
                y_to_plot = np.linspace(low_y_bound, high_y_bound / 1.02, 200)
                xv, yv = np.meshgrid(x_to_plot, y_to_plot)
                grid_points = np.stack([xv.flatten(), yv.flatten()], axis=-1)

                mult_observations = jnp.tile(ob[None], (grid_points.shape[0],1))
                mult_observations = mult_observations.at[:, :2].set(grid_points)
                goal_xy_mult = jnp.tile((goal_xy)[None], (grid_points.shape[0],1))

                rng = jax.random.PRNGKey(0)
                curr_rng, rng = jax.random.split(rng)

                value_loss_eval, eval_info = eval_step(
                    network, agent, mult_observations, goal_xy_mult, grid_points, curr_rng, config['critic_loss_type']
                )
                value_loss_eval = value_loss_eval.item()

                for k, v in eval_info.items():
                    if k != 'pred' and k != 'q_pred' and k != 'value_loss_per_point':
                        eval_metrics[f'evaluation/task_{task_id}_{k}'] = v

                pred = eval_info['pred'].mean(axis=0)
                q_pred = eval_info['q_pred'].mean(axis=0)
                value_loss_per_point = eval_info['value_loss_per_point'].mean(axis=0)

                plt.clf()
                plt.scatter(grid_points[:,0], grid_points[:,1], c=pred, s=1, cmap='plasma')
                plt.colorbar()
                plt.scatter(all_cells[:,0], all_cells[:,1], c='gray', s=10)
                plt.scatter(goal_ob[0], goal_ob[1], c='red', s=50, marker='*')
                plt.savefig(f'{FLAGS.save_dir}/task_{task_id}_value.png', dpi=300)

                wandb.log({f'evaluation/task_{task_id}_value': wandb.Image(f'{FLAGS.save_dir}/task_{task_id}_value.png')}, step=i)
                os.remove(f'{FLAGS.save_dir}/task_{task_id}_value.png')

                plt.clf()
                plt.scatter(grid_points[:,0], grid_points[:,1], c=q_pred, s=1, cmap='plasma')
                plt.colorbar()
                plt.scatter(all_cells[:,0], all_cells[:,1], c='gray', s=10)
                plt.scatter(goal_ob[0], goal_ob[1], c='red', s=50, marker='*')
                plt.savefig(f'{FLAGS.save_dir}/task_{task_id}_q_pred.png', dpi=300)
                
                wandb.log({f'evaluation/task_{task_id}_q_pred': wandb.Image(f'{FLAGS.save_dir}/task_{task_id}_q_pred.png')}, step=i)
                os.remove(f'{FLAGS.save_dir}/task_{task_id}_q_pred.png')

                plt.clf()
                plt.scatter(grid_points[:,0], grid_points[:,1], c=value_loss_per_point, s=1, cmap='plasma')
                plt.colorbar()
                plt.scatter(all_cells[:,0], all_cells[:,1], c='gray', s=10)
                plt.scatter(goal_ob[0], goal_ob[1], c='red', s=50, marker='*')
                plt.savefig(f'{FLAGS.save_dir}/task_{task_id}_abs_diff.png', dpi=300)
                wandb.log({f'evaluation/task_{task_id}_abs_diff': wandb.Image(f'{FLAGS.save_dir}/task_{task_id}_abs_diff.png')}, step=i)
                os.remove(f'{FLAGS.save_dir}/task_{task_id}_abs_diff.png')

                wandb.log(eval_metrics, step=i)
                
                # time_elapsed = time.time() - start_time
                # print(f'Evaluated task {task_id} after {time_elapsed:.2f} seconds.', file=sys.stderr)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(network, FLAGS.save_dir, i)


    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)