import glob
import json
import os
import pathlib
import random
import signal
import sys
import time

import jax
import numpy as np
import tqdm
import wandb
import wandb.util
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from env_wrappers import MazeEnvWrapper
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_animal, get_exp_name, get_flag_dict, setup_wandb
from utils.plot_utils import plot_data
from utils.samplers import to_oracle_rep
from utils.statistics import statistics
from wrappers import wrappers
from wrappers.datafuncs_utils import clip_dataset, make_env_and_datasets

FLAGS = flags.FLAGS

##=========== WANDB SPECIFICATION ===========##
flags.DEFINE_string('wbproj', 'aorl', 'Weights & Biases project name.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('wandb_alerts', True, 'Enable Weights & Biases alerts.')

##=========== ENVIRONMENT SPECIFICATION ===========##
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_integer('train_data_size', 1000000, 'Size of training data to use (None for full dataset).')

##=========== AGENT SPECIFICATION ===========##
config_flags.DEFINE_config_file('agent', 'agents/sharsa.py', lock_config=False)
flags.DEFINE_string('save_dir', '../../scratch', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

##=========== TRAINING HYPERPARAMETERS ===========##
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')
flags.DEFINE_integer('collection_steps', 1000000, 'Number of data collection steps.')
flags.DEFINE_integer('data_plot_interval', 100000, 'Data plotting interval.')
flags.DEFINE_bool('cleanup', False, 'If true, delete saved data and weight checkpoints at run end.')

##=========== EVALUATION HYPERPARAMETERS ===========##
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

##=========== DATA COLLECTION FLAGS ===========##
config_flags.DEFINE_config_file('wrapper', None, lock_config=False)
##=========== END FLAGS ===========##

PREEMPTED = {'flag': False}


def _dataset_mapping(dataset):
    return dataset.dataset if hasattr(dataset, 'dataset') else dataset


def _load_npz_dict(path):
    file = np.load(path)
    return {k: file[k][...] for k in file.files}


def _write_global_step_atomically(save_dir, global_step):
    tmp = pathlib.Path(save_dir) / 'global_step.tmp'
    final = pathlib.Path(save_dir) / 'global_step'
    tmp.write_text(str(global_step))
    tmp.replace(final)


def _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step, global_step_file=None):
    save_agent(agent, save_dir, global_step)
    np.savez(os.path.join(save_dir, f'data-{global_step}.npz'), **_dataset_mapping(train_dataset))
    np.savez(os.path.join(save_dir, f'data-{global_step}-val.npz'), **_dataset_mapping(val_dataset))
    if global_step_file is not None:
        global_step_file.write_text(str(global_step))


def _cleanup_checkpoints(save_dir):
    removed = []
    for pattern in ('params_*.pkl', 'data-*.npz'):
        for path in pathlib.Path(save_dir).glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print(f'Cleanup enabled. Removed {len(removed)} checkpoint files.')


def _get_train_dataset_size(train_dataset):
    if hasattr(train_dataset, 'size'):
        return int(train_dataset.size)

    dataset = _dataset_mapping(train_dataset)
    if isinstance(dataset, dict):
        for key in ('observations', 'actions', 'rewards', 'terminals', 'next_observations'):
            if key in dataset:
                return int(len(dataset[key]))
        if len(dataset) > 0:
            first_value = next(iter(dataset.values()))
            return int(len(first_value))

    try:
        return int(len(train_dataset))
    except TypeError:
        return None


def _log_prefixed_info(metrics, prefix, global_step):
    for k, v in metrics.items():
        wandb.log({f'{prefix}/{k}': v}, step=global_step)


def _maybe_checkpoint(agent, train_dataset, val_dataset, save_dir, global_step, train_logger, eval_logger):
    if PREEMPTED['flag']:
        checkpoint_and_exit(
            agent,
            train_dataset,
            val_dataset,
            save_dir,
            global_step,
            train_logger,
            eval_logger,
            reason='signal',
        )


def _evaluate(agent, config, env):
    return env.evaluate_step(
        agent,
        config,
        env_name=FLAGS.env_name,
        eval_episodes=FLAGS.eval_episodes,
        video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
        eval_temperature=FLAGS.eval_temperature,
        eval_gaussian=FLAGS.eval_gaussian,
    )


def _train_step(
    *,
    agent,
    train_dataset,
    val_dataset,
    config,
    env,
    global_step,
    pbar,
    first_time,
    last_time,
    train_logger,
    eval_logger,
    save_dir,
    global_step_file,
):
    batch = train_dataset.sample(config['batch_size'])
    agent, update_info = agent.update(batch)
    global_step += 1
    pbar.update(1)

    _maybe_checkpoint(agent, train_dataset, val_dataset, save_dir, global_step, train_logger, eval_logger)

    if global_step % FLAGS.log_interval == 0:
        train_metrics = {f'training/{k}': v for k, v in update_info.items()}
        val_batch = val_dataset.sample(config['batch_size'])
        _, val_info = agent.total_loss(val_batch, grad_params=None)
        train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

        train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
        train_metrics['time/total_time'] = time.time() - first_time
        last_time = time.time()
        wandb.log(train_metrics, step=global_step)
        train_logger.log(train_metrics, step=global_step)

    if FLAGS.eval_interval != 0 and (global_step == 1 or global_step % FLAGS.eval_interval == 0):
        eval_metrics = _evaluate(agent, config, env)
        wandb.log(eval_metrics, step=global_step)
        eval_logger.log(eval_metrics, step=global_step)

    if global_step % FLAGS.save_interval == 0:
        _save_agent_and_datasets(
            agent,
            train_dataset,
            val_dataset,
            save_dir,
            global_step,
            global_step_file=global_step_file,
        )

    return agent, global_step, last_time


def handle_preempt(signum, frame):
    del signum, frame
    PREEMPTED['flag'] = True
    print('Received preemption signal. Will save and exit after current epoch.', file=sys.stderr)


for sig in (signal.SIGUSR1, signal.SIGTERM):
    signal.signal(sig, handle_preempt)


def checkpoint_and_exit(
    agent,
    train_dataset,
    val_dataset,
    save_dir,
    global_step,
    train_logger=None,
    eval_logger=None,
    *,
    reason='preempt',
):
    _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step)

    if train_logger:
        train_logger.close()
    if eval_logger:
        eval_logger.close()

    _write_global_step_atomically(save_dir, global_step)

    try:
        wandb.alert(title='Preempted', text=f'Checkpointed at step {global_step} ({reason})')
    except Exception:
        pass
    try:
        wandb.finish()
    except Exception:
        pass
    raise SystemExit(0)


##=========== MAIN SCRIPT ===========##
def main(_):
    assert 'humanoidmaze' not in FLAGS.env_name or FLAGS.agent['discount'] == 0.995, (
        'Humanoid maze tasks require discount factor of 0.995.'
    )
    assert FLAGS.dataset_dir is not None, 'must provide dataset directory'
    assert FLAGS.agent['actor_type'] == 'best-of-n', 'evaluation only implemented for best-of-n actors'

    exp_name, info = get_exp_name(seed=FLAGS.seed, config=FLAGS)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wbproj, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    wandb_id_file = pathlib.Path(FLAGS.save_dir) / 'wandb_id'
    if wandb_id_file.exists():
        wbid = wandb_id_file.read_text().strip()
        resume = 'allow'
    else:
        wbid = wandb.util.generate_id()
        wandb_id_file.write_text(wbid)
        resume = None

    setup_wandb(project='aorl', entity='moma1234', group=FLAGS.run_group, name=exp_name, id=wbid, resume=resume)
    wandb.run.config.update({'info': info}, allow_val_change=True)
    print(f'Created new run {wandb.run.name} with ID {wandb.run.id}')

    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    if FLAGS.wandb_alerts:
        animal = get_animal()
        print(f'\n{animal}\n', exp_name)
        print('\n\n', info, file=sys.stderr)
        print('\n\npython', ' '.join(sys.argv), '\n', file=sys.stderr)
        wandb.run.alert(
            title=f'{animal} e2e run started!',
            text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}",
        )

    total_steps = 2 * FLAGS.offline_steps + FLAGS.collection_steps
    dataset_class_dict = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }
    config = FLAGS.agent

    global_step_file = pathlib.Path(FLAGS.save_dir) / 'global_step'
    train_logger = None
    eval_logger = None

    if global_step_file.exists() and int(global_step_file.read_text().strip()) > 0:
        global_step = int(global_step_file.read_text().strip())
        print(f'Restoring from epoch {global_step}')

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        dataset_path = pathlib.Path(FLAGS.save_dir) / f'data-{global_step}.npz'
        env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=str(dataset_path), use_oracle_reps=True)
        # env = MazeEnvWrapper(env, seed=FLAGS.seed)
        # want the randomly generated tasks to be the same each time, so I won't pass in the seed - let it be 0
        env = MazeEnvWrapper(env)

        train_dataset_data = _load_npz_dict(dataset_path)
        val_dataset_data = _load_npz_dict(pathlib.Path(FLAGS.save_dir) / f'data-{global_step}-val.npz')

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        print(f'Evaluating on {len(env.task_infos)} tasks with start_ij {env.start_ij}')

        dataset_class = dataset_class_dict[config['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset_data, freeze=False), config)
        val_dataset = dataset_class(Dataset.create(**val_dataset_data, freeze=False), config)
        example_batch = train_dataset.sample(1)

        agent_class = agents[config['agent_name']]
        agent = agent_class.create(FLAGS.seed, example_batch, config)
        print(agent.config, file=sys.stderr)
        agent = restore_agent(agent, FLAGS.save_dir, global_step)

        data_collection_env = None
        if global_step < FLAGS.offline_steps or FLAGS.offline_steps + FLAGS.collection_steps <= global_step:
            train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
            eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
            first_time = time.time()
            last_time = time.time()
    else:
        global_step = 0
        global_step_file.write_text(str(global_step))

        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
        dataset_idx = 0
        env, train_dataset_data, val_dataset_data = make_env_and_datasets(
            FLAGS.env_name,
            dataset_path=datasets[dataset_idx],
            use_oracle_reps=True,
        )
        # env = MazeEnvWrapper(env, seed=FLAGS.seed)
        # same thing, let the seed just be 0
        env = MazeEnvWrapper(env)

        train_dataset_data = clip_dataset(train_dataset_data, int(FLAGS.train_data_size))

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        print(f'Evaluating on {len(env.task_infos)} tasks with start_ij {env.start_ij}')

        dataset_class = dataset_class_dict[config['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset_data, freeze=False), config)
        val_dataset = dataset_class(Dataset.create(**val_dataset_data, freeze=False), config)
        example_batch = train_dataset.sample(1)

        agent_class = agents[config['agent_name']]
        agent = agent_class.create(FLAGS.seed, example_batch, config)
        print(agent.config, file=sys.stderr)

        if FLAGS.restore_path is not None:
            agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        data_collection_env = None

    with tqdm.tqdm(total=total_steps, initial=global_step) as pbar:
        while global_step < total_steps:
            if global_step < FLAGS.offline_steps:
                if global_step == 0:
                    task_info_to_plot = {
                        'start_xy': {'x': [env.start_xy[0]], 'y': [env.start_xy[1]], 's': 50, 'c': 'red'},
                        'all_cells': {'x': env.all_cells[:, 0], 'y': env.all_cells[:, 1], 's': 1, 'c': 'lightgrey'},
                    }
                    for t in env.task_infos:
                        task_info_to_plot[t['task_name']] = {
                            'x': t['goal_xy'][0],
                            'y': t['goal_xy'][1],
                            's': 50,
                            'c': random.choice(['blue', 'green', 'orange', 'purple', 'brown']),
                            'marker': random.choice(['*', 'X', 'P', 'D', 'v']),
                        }
                    fig_name = plot_data(task_info_to_plot, save_dir=FLAGS.save_dir)
                    wandb.log({'data_collection/task_info_viz': wandb.Image(fig_name)})
                    print(f'Plotted task info to {fig_name}')
                    os.remove(fig_name)

                    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
                    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
                    first_time = time.time()
                    last_time = time.time()
                    print(f'Beginning training for {FLAGS.offline_steps} steps', file=sys.stderr)

                agent, global_step, last_time = _train_step(
                    agent=agent,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    config=config,
                    env=env,
                    global_step=global_step,
                    pbar=pbar,
                    first_time=first_time,
                    last_time=last_time,
                    train_logger=train_logger,
                    eval_logger=eval_logger,
                    save_dir=FLAGS.save_dir,
                    global_step_file=global_step_file,
                )

                if global_step == FLAGS.offline_steps:
                    train_logger.close()
                    eval_logger.close()

            if FLAGS.offline_steps <= global_step < FLAGS.offline_steps + FLAGS.collection_steps:
                if data_collection_env is None:
                    data_collection_env = make_env_and_datasets(
                        FLAGS.env_name,
                        dataset_path='',
                        env_only=True,
                        use_oracle_reps=True,
                        terminate_at_goal=False,
                        max_episode_steps=FLAGS.wrapper.get('max_episode_steps', 2000),
                    )

                    wrapper = wrappers[FLAGS.wrapper['method_name']]
                    clipped_dataset = clip_dataset(train_dataset.dataset, FLAGS.train_data_size)
                    collection_agent = wrapper.create(
                        agent=agent,
                        train_dataset=clipped_dataset,
                        config=FLAGS.wrapper,
                    )

                    rbsize = FLAGS.train_data_size + FLAGS.collection_steps
                    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset.dataset), rbsize)
                    rng = jax.random.PRNGKey(FLAGS.seed)

                    num_additional = global_step - FLAGS.offline_steps
                    assert FLAGS.train_data_size + num_additional < rbsize
                    train_dataset.pointer = FLAGS.train_data_size + num_additional
                    train_dataset.size = train_dataset.pointer

                    ob, _ = data_collection_env.reset(
                        options=dict(task_info=dict(init_ij=env.start_ij, goal_ij=env.start_ij))
                    )
                    collection_agent, pre_info = collection_agent.pre(observations=ob, rng=rng)
                    _log_prefixed_info(pre_info, 'data_collection/pre', global_step)

                    goal_xy = collection_agent.curr_goal
                    goal_ij = data_collection_env.unwrapped.xy_to_ij(goal_xy)
                    ob, _ = data_collection_env.reset(
                        options=dict(task_info=dict(init_ij=env.start_ij, goal_ij=goal_ij))
                    )
                    done = False

                    all_cells = env.all_cells
                    vertex_cells = env.vertex_cells
                    data_to_plot = {
                        'all_cells': {'x': all_cells[:, 0], 'y': all_cells[:, 1], 's': 1, 'c': 'lightgrey'},
                        'vertex_cells': {'x': vertex_cells[:, 0], 'y': vertex_cells[:, 1], 's': 5, 'c': 'grey'},
                        'buffer': {'x': [], 'y': [], 's': 1, 'c': [], 'cmap': 'plasma'},
                        'goals': {
                            'x': [goal_xy[0]],
                            'y': [goal_xy[1]],
                            's': 50,
                            'c': [global_step],
                            'cmap': 'viridis',
                            'marker': '*',
                        },
                    }

                    stats = statistics[FLAGS.env_name](env=data_collection_env)
                else:
                    collection_agent, pre_info = collection_agent.pre(observations=ob, rng=rng)
                    _log_prefixed_info(pre_info, 'data_collection/pre', global_step)

                curr_rng, rng = jax.random.split(rng)
                action, _ = collection_agent.sample_actions(
                    observations=ob,
                    goals=None,
                    seed=curr_rng,
                    pre_info=pre_info,
                )
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = data_collection_env.step(action)
                done = terminated or truncated

                tran = dict(
                    observations=ob,
                    actions=action,
                    terminals=float(done),
                    next_observations=next_ob,
                    qpos=info['qpos'],
                    qvel=info['qvel'],
                    oracle_reps=to_oracle_rep(obs=ob[None], env=env)[0],
                )

                train_dataset.add_transition(tran)
                data_to_plot['buffer']['x'].append(tran['oracle_reps'][0])
                data_to_plot['buffer']['y'].append(tran['oracle_reps'][1])
                data_to_plot['buffer']['c'].append(global_step)
                stats.log_episode(tran['observations'], tran['actions'])

                collection_agent, post_info = collection_agent.post(transition=tran, rng=rng)
                _log_prefixed_info(post_info, 'data_collection/post', global_step)

                global_step += 1
                pbar.update(1)
                _maybe_checkpoint(
                    agent,
                    train_dataset,
                    val_dataset,
                    FLAGS.save_dir,
                    global_step,
                    train_logger,
                    eval_logger,
                )

                if done:
                    goal_xy = collection_agent.curr_goal
                    goal_ij = data_collection_env.unwrapped.xy_to_ij(goal_xy)
                    ob, _ = data_collection_env.reset(
                        options=dict(task_info=dict(init_ij=env.start_ij, goal_ij=goal_ij))
                    )
                    data_to_plot['goals']['x'].append(goal_xy[0])
                    data_to_plot['goals']['y'].append(goal_xy[1])
                    data_to_plot['goals']['c'].append(global_step)
                else:
                    ob = next_ob

                if global_step % FLAGS.log_interval == 0:
                    for k, v in stats.get_statistics().items():
                        wandb.log({f'data_collection/{k}': v}, step=global_step)

                if global_step % FLAGS.data_plot_interval == 0:
                    fig_name = plot_data(data_to_plot, save_dir=FLAGS.save_dir)
                    wandb.log({'data_collection/data_viz': wandb.Image(fig_name)}, step=global_step)
                    print(f'Plotted data to {fig_name}')
                    os.remove(fig_name)

                if global_step % FLAGS.save_interval == 0:
                    _save_agent_and_datasets(
                        agent,
                        train_dataset,
                        val_dataset,
                        FLAGS.save_dir,
                        global_step,
                        global_step_file=global_step_file,
                    )

                if global_step == FLAGS.offline_steps + FLAGS.collection_steps:
                    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
                    print(f'new replay buffer size: {train_dataset.size}')

            if FLAGS.offline_steps + FLAGS.collection_steps <= global_step:
                if global_step == FLAGS.offline_steps + FLAGS.collection_steps:
                    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
                    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))
                    first_time = time.time()
                    last_time = time.time()
                    print(f'Beginning further training for {FLAGS.offline_steps} steps', file=sys.stderr)

                agent, global_step, last_time = _train_step(
                    agent=agent,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    config=config,
                    env=env,
                    global_step=global_step,
                    pbar=pbar,
                    first_time=first_time,
                    last_time=last_time,
                    train_logger=train_logger,
                    eval_logger=eval_logger,
                    save_dir=FLAGS.save_dir,
                    global_step_file=global_step_file,
                )

                if global_step == total_steps:
                    train_logger.close()
                    eval_logger.close()
                    if FLAGS.cleanup:
                        _cleanup_checkpoints(FLAGS.save_dir)

            run_metrics = {'global_step': global_step}
            train_dataset_size = _get_train_dataset_size(train_dataset)
            if train_dataset_size is not None:
                run_metrics['data/train_dataset_size'] = train_dataset_size
            wandb.log(run_metrics)


if __name__ == '__main__':
    app.run(main)
