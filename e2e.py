import glob
import json
import os
import random
import time
from collections import defaultdict
import sys

import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer
from utils.evaluation import evaluate_gcfql, evaluate_custom_gcfql
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb, get_animal
from datafuncs import datafuncs

import numpy as np
import ogbench

from utils.datasets import Dataset
from datafuncs.datafuncs_utils import clip_dataset, make_env_and_datasets
from utils.plot_utils import plot_data

FLAGS = flags.FLAGS

##=========== WANDB SPECIFICATION ===========##
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('wandb_alerts', True, 'Enable Weights & Biases alerts.')

##=========== ENVIRONMENT SPECIFICATION ===========##
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('train_data_size', None, 'Size of training data to use (None for full dataset).')

##=========== AGENT SPECIFICATION ===========##
config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

##=========== TRAINING HYPERPARAMETERS ===========##
flags.DEFINE_integer('offline_steps', 2000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 100000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

##=========== EVALUATION HYPERPARAMETERS ===========##
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

##=========== DATA COLLECTION FLAGS ===========##
config_flags.DEFINE_config_file('data_option', None, 'Data function option (e.g., new_buffer, combine_with).')
flags.DEFINE_bool('debug', False, 'Debug mode.')
flags.DEFINE_string('wbid', None, 'Weights & Biases ID (for resuming runs).')

def print_info(exp_name, info):
    animal = get_animal()
    print(f"\n{animal}\n", exp_name)
    print("\n\n", info, file=sys.stderr)
    print("\n\npython", " ".join(sys.argv), "\n", file=sys.stderr)
    return animal

def to_jnp(batch):
    return jax.tree_util.tree_map(lambda x: jnp.array(x), batch)
    # return batch

def choose_start_ij(env):
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

                # Exclude hallway cells.
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))
                
    init_ij = vertex_cells[np.random.randint(len(vertex_cells))]
    init_xy = env.unwrapped.ij_to_xy(init_ij)
    return {'init_ij': init_ij, 'init_xy': init_xy}

def create_task_infos(env, start_ij):
    NUM_TASKS = 5
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

                # Exclude hallway cells.
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))
    idxs = np.argwhere(maze_map == 1)
    task_info = []
    # for task_i in range(1, NUM_TASKS + 1):
    for task_i, (i, j) in enumerate(vertex_cells):
        # i, j = vertex_cells[np.random.randint(len(vertex_cells))]
        task_info.append({
            'task_name': f'custom_task{task_i}',
            'init_ij': start_ij,
            'goal_ij': (i, j),
            'goal_xy': env.unwrapped.ij_to_xy((i, j)),
        })

    return task_info

##=========== MAIN SCRIPT ===========##
def main(_):
    ##=========== ASSERTIONS ===========##
    if 'humanoidmaze' in FLAGS.env_name:
            assert FLAGS.agent['discount'] == 0.995, "Humanoid maze tasks require discount factor of 0.995."
    assert FLAGS.dataset_dir is not None, 'must provide dataset directory'

    def evaluate_step(agent, env, config, task_info=None):
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        if task_info is not None:
            task_infos = task_info
        else:
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = len(task_infos)
        for task_id in tqdm.trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = evaluate_custom_gcfql(
                agent=agent,
                env=env,
                env_name=FLAGS.env_name,
                goal_conditioned=True,
                # task_id=task_id,
                init_ij=task_infos[task_id - 1]['init_ij'],
                goal_ij=task_infos[task_id - 1]['goal_ij'],
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)
        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=5)
                eval_metrics['video'] = video

        return eval_metrics


    # Set up logger.
    exp_name, info = get_exp_name(FLAGS.seed, config=FLAGS)

    if FLAGS.wbid is not None:
        api = wandb.Api()

        try:
            run = api.run(f"jnzhao3/aorl/{FLAGS.wbid}")
            exp_name = run.name
            print(f"Resuming run {run.name} with ID {run.id}")
        except Exception as e:
            print(f"Failed to find run with ID {FLAGS.wbid}, starting new run")
            run = None
        

        setup_wandb(project='aorl', group=FLAGS.run_group, name=exp_name, id=FLAGS.wbid)
        FLAGS.wbid = wandb.run.id
        wandb.run.config.update({'info': info}, allow_val_change=True)
        print(f"Created new run {wandb.run.name} with ID {wandb.run.id}")
    else:
        setup_wandb(project='aorl', group=FLAGS.run_group, name=exp_name)
        FLAGS.wbid = wandb.run.id
        wandb.run.config.update({'info': info}, allow_val_change=True)
        # PREEMPTED = False

    ##=========== LOG MESSAGES TO ERR AND SLACK ===========##
    animal = print_info(exp_name, info)
    
    if FLAGS.wandb_alerts:
        wandb.run.alert(title=f"{animal} e2e run started!", text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}")

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and datasets.
    config = FLAGS.agent
    datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
    dataset_idx = 0
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx], use_oracle_reps=True)

    N = int(FLAGS.train_data_size)
    train_dataset = clip_dataset(train_dataset, N)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }

    dataset_class = dataset_class_dict[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset, freeze=False), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset, freeze=False), config)
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )
    print(agent.config, file=sys.stderr)

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    elif '_checkpoint_epoch' in dict(wandb.run.summary):
        restore_epoch = int(wandb.run.summary['_checkpoint_epoch'])
        print(f"Restoring from epoch {restore_epoch}")
        agent = restore_agent(agent, FLAGS.save_dir, restore_epoch)

    assert agent.config['actor_type'] == 'best-of-n', "evaluation only implemented for best-of-n actors"
    ##=========== SET EVALUATION INFO ===========##
    start_ij = choose_start_ij(env)['init_ij']
    task_info = create_task_infos(env, start_ij=start_ij)
    # env.task_infos = task_info
    print(f"Evaluating on {len(task_info)} tasks with start_ij {start_ij}")

    ##=========== PLOT THE TASK INFOS ===========##
    task_info_to_plot = {'start_xy' : {
        'x': [env.unwrapped.ij_to_xy(start_ij)[0]],
        'y': [env.unwrapped.ij_to_xy(start_ij)[1]],
        's': 50,'c': 'red',
    }}
    for t in task_info:
        task_info_to_plot[t['task_name']] = {
            'x': t['goal_xy'][0],
            'y': t['goal_xy'][1],
            's': 50,
            'c': random.choice(['blue', 'green', 'orange', 'purple', 'brown']),
            'marker': random.choice(['*', 'X', 'P', 'D', 'v']),
        }
    fig_name = plot_data(
        task_info_to_plot,
        save_dir=FLAGS.save_dir,
    )
    wandb.log({"data_collection/task_info_viz": wandb.Image(fig_name)})
    print(f"Plotted task info to {fig_name}")
    os.remove(fig_name)


    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    if '_checkpoint_epoch' in dict(wandb.run.summary):
        start_i = int(wandb.run.summary['_checkpoint_epoch']) + 1
    else:
        start_i = 1
    for i in tqdm.tqdm(range(start_i, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if FLAGS.debug:
            break
        # batch = to_jnp(train_dataset.sample(config['batch_size']))
        batch = train_dataset.sample(config['batch_size'])
        batch = to_jnp(batch)
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            # val_batch = to_jnp(val_dataset.sample(config['batch_size']))
            val_batch = val_dataset.sample(config['batch_size'])
            val_batch = to_jnp(val_batch)
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = evaluate_step(agent, env, config, task_info=task_info)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            wandb.run.summary['_checkpoint_epoch'] = i

    train_logger.close()
    eval_logger.close()

    ##=========== ADD NEW DATA ===========##
    datafunc = datafuncs.get(FLAGS.data_option['method_name'], None)
    assert datafunc is not None, f'unknown data option {FLAGS.data_option}'
    replay_buffer = datafunc.create(original_dataset=train_dataset, config=FLAGS.data_option, env=env, agent_config=config, seed=FLAGS.seed, save_dir=FLAGS.save_dir, start_ij=start_ij, wandb=wandb, agent=agent, train_dataset=train_dataset)

    replay_buffer = dataset_class(Dataset.create(**replay_buffer), config)
    # val_dataset = dataset_class(Dataset.create(**replay_buffer, freeze=False), config)
    # original_dataset, config, agent_config, env, seed, save_dir
    print(f'new replay buffer size: {replay_buffer.size}')

    ##=========== FURTHER TRAINING ===========##
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))
    first_time = time.time()
    last_time = time.time()

    # if PREEMPTED and wandb.run.summary.get('_further_checkpoint_epoch', 0) > 0:
    if '_further_checkpoint_epoch' in dict(wandb.run.summary):
        agent = restore_agent(agent, FLAGS.save_dir, int(wandb.run.summary['_further_checkpoint_epoch']))
        start_i = int(wandb.run.summary['_further_checkpoint_epoch']) + 1
    else:
        start_i = 1

    for i in tqdm.tqdm(range(start_i, 2 * FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = replay_buffer.sample(config['batch_size'])
        batch = to_jnp(batch)
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            val_batch = val_dataset.sample(config['batch_size'])
            val_batch = to_jnp(val_batch)
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = evaluate_step(agent, env, config, task_info=task_info)

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            wandb.run.summary['_further_checkpoint_epoch'] = i

    train_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    app.run(main)