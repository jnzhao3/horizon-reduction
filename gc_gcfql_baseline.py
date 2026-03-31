import glob
import json
import os
import pathlib
import random
import signal
import sys
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
import wandb.util
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_animal, get_exp_name, get_flag_dict, setup_wandb
from utils.plot_utils import plot_heatmap
from utils.samplers import to_oracle_rep
from utils.statistics import get_statistics_class
from wrappers.datafuncs_utils import clip_dataset, make_env_and_datasets
from utils.evaluation import evaluate_gcfql, evaluate
from ogbench.relabel_utils import add_oracle_reps, relabel_dataset

FLAGS = flags.FLAGS

##=========== WANDB SPECIFICATION ===========##
flags.DEFINE_string('wbproj', 'aorl2', 'Weights & Biases project name.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('wandb_alerts', True, 'Enable Weights & Biases alerts.')

##=========== ENVIRONMENT SPECIFICATION ===========##
flags.DEFINE_string('env_name', 'puzzle-4x5-play-oraclerep-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')

##=========== AGENT SPECIFICATION ===========##
config_flags.DEFINE_config_file('agent', 'agents/gcfql.py', lock_config=False)
config_flags.DEFINE_config_file('fql_agent', 'agents/fql.py', lock_config=False)
flags.DEFINE_string('save_dir', '../../scratch', 'Save directory.')

##=========== TRAINING HYPERPARAMETERS ===========##
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('further_offline_steps', 1000000, 'Number of offline steps for the second round of training') # TODO: eventually, delete this
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
##=========== END FLAGS ===========##

PREEMPTED = {'flag': False}


def _dataset_mapping(dataset):
    if hasattr(dataset, 'dataset'):
        return dataset.dataset
    if hasattr(dataset, '_dict'):
        return dataset._dict
    return dataset


def _load_npz_dict(path):
    file = np.load(path)
    return {k: file[k][...] for k in file.files}


def _write_global_step_atomically(save_dir, global_step):
    tmp = pathlib.Path(save_dir) / 'global_step.tmp'
    final = pathlib.Path(save_dir) / 'global_step'
    tmp.write_text(str(global_step))
    tmp.replace(final)


def _write_int_atomically(save_dir, name, value):
    tmp = pathlib.Path(save_dir) / f'{name}.tmp'
    final = pathlib.Path(save_dir) / name
    tmp.write_text(str(int(value)))
    tmp.replace(final)


def _write_replaybuffer_state_atomically(save_dir, train_dataset):
    if not isinstance(train_dataset, ReplayBuffer):
        return
    _write_int_atomically(save_dir, 'rbsize', train_dataset.max_size)
    _write_int_atomically(save_dir, 'replaybuffer_pointer', train_dataset.pointer)


def _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step, global_step_file=None, collection_state=None):
    save_agent(agent, save_dir, global_step)
    np.savez(os.path.join(save_dir, f'data-{global_step}.npz'), **_dataset_mapping(train_dataset))
    np.savez(os.path.join(save_dir, f'data-{global_step}-val.npz'), **_dataset_mapping(val_dataset))
    if global_step_file is not None:
        global_step_file.write_text(str(global_step))
    if isinstance(train_dataset, ReplayBuffer):
        pathlib.Path(save_dir, 'rbsize').write_text(str(train_dataset.max_size))
        pathlib.Path(save_dir, 'replaybuffer_pointer').write_text(str(train_dataset.pointer))
    if collection_state is not None:
        import pickle
        collection_state_to_save = collection_state.copy()
        collection_state_to_save['env'] = None  # Don't save env
        with open(os.path.join(save_dir, f'collection_state-{global_step}.pkl'), 'wb') as f:
            pickle.dump(collection_state_to_save, f)


def _cleanup_checkpoints(save_dir):
    removed = []
    for pattern in ('params_*.pkl', 'data-*.npz', 'collection_state-*.pkl'):
        for path in pathlib.Path(save_dir).glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print(f'Cleanup enabled. Removed {len(removed)} checkpoint files.')


def _create_agent(agent_name, seed, example_batch, config):
    agent_class = agents[agent_name]
    if agent_name == 'fql':
        return agent_class.create(seed, example_batch['observations'], example_batch['actions'], config)
    return agent_class.create(seed, example_batch, config)


def _wrap_goal_conditioned_dataset(dataset, config):
    if isinstance(dataset, GCDataset):
        return dataset
    if isinstance(dataset, ReplayBuffer):
        return dataset
    return GCDataset(dataset, config)


def _maybe_checkpoint(agent, train_dataset, val_dataset, save_dir, global_step, train_logger, eval_logger, collection_state=None):
    if PREEMPTED['flag']:
        checkpoint_and_exit(
            agent,
            train_dataset,
            val_dataset,
            save_dir,
            global_step,
            train_logger,
            eval_logger,
            collection_state=collection_state,
            reason='signal',
        )


def _evaluate(agent, config, env, prefix=''):

    if agent.config['agent_name'] == 'fql':
        eval_info, _, _ = evaluate(
                    agent=agent,
                    env=env,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
    
        if prefix != '':
            new_eval_info = {}
            for k, v in eval_info.items():
                new_eval_info[f'{prefix}{k}'] = v
            eval_info = new_eval_info
        return eval_info
    
    else:

        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        eval_metrics = {}
        overall_metrics = defaultdict(list)

        for task_id, task_info in enumerate(task_infos, start=1):
            task_name = task_info.get('task_name', f'task{task_id}')
            eval_info, _, _ = evaluate_gcfql(
                agent=agent,
                env=env,
                env_name=FLAGS.env_name,
                goal_conditioned=True,
                task_id=task_id,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
                use_oracle_rep=True
            )
            for metric_name, metric_value in eval_info.items():
                eval_metrics[f'{prefix}evaluation/{task_name}_{metric_name}'] = metric_value
                overall_metrics[metric_name].append(metric_value)

        for metric_name, values in overall_metrics.items():
            eval_metrics[f'{prefix}evaluation/overall_{metric_name}'] = np.mean(values)

        return eval_metrics


def _get_env_current_task(env):
    candidates = [env]
    if hasattr(env, 'unwrapped'):
        candidates.append(env.unwrapped)

    for candidate in candidates:
        cur_task = getattr(candidate, 'cur_task', None)
        if cur_task is not None:
            return cur_task

    for candidate in candidates:
        cur_task_id = getattr(candidate, 'cur_task_id', None)
        task_infos = getattr(candidate, 'task_infos', None)
        if cur_task_id is not None and task_infos is not None:
            return task_infos[cur_task_id - 1]

    return None


def _get_task_goal(env):
    cur_task = _get_env_current_task(env)
    if cur_task is None:
        raise ValueError('Environment does not expose a current task goal.')

    for key in ('goal', 'goal_xy', 'goal_oracle_rep'):
        if key in cur_task and cur_task[key] is not None:
            goal = np.asarray(cur_task[key])
            if goal.ndim > 1:
                goal = goal[0]
            return goal

    if 'goal_ij' in cur_task and hasattr(env.unwrapped, 'ij_to_xy'):
        return np.asarray(env.unwrapped.ij_to_xy(cur_task['goal_ij']))

    raise ValueError('Current task does not expose a usable fixed goal.')


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
    prefix='',
):
    batch = train_dataset.sample(config['batch_size'])
    agent, update_info = agent.update(batch)
    global_step += 1
    pbar.update(1)

    _maybe_checkpoint(agent, train_dataset, val_dataset, save_dir, global_step, train_logger, eval_logger)

    if global_step % FLAGS.log_interval == 0:
        train_metrics = {f'{prefix}training/{k}': v for k, v in update_info.items()}
        val_batch = val_dataset.sample(config['batch_size'])
        _, val_info = agent.total_loss(val_batch, grad_params=None)
        train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

        train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
        train_metrics['time/total_time'] = time.time() - first_time
        last_time = time.time()
        wandb.log(train_metrics, step=global_step)
        train_logger.log(train_metrics, step=global_step)

    if FLAGS.eval_interval != 0 and (global_step == 1 or global_step % FLAGS.eval_interval == 0):
        eval_metrics = _evaluate(agent, config, env, prefix=prefix)
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
    collection_state=None,
    *,
    reason='preempt',
):
    _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step, collection_state=collection_state)

    if train_logger:
        train_logger.close()
    if eval_logger:
        eval_logger.close()

    _write_global_step_atomically(save_dir, global_step)
    _write_replaybuffer_state_atomically(save_dir, train_dataset)

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

    ##=========== SET-UP ===========##
    assert 'humanoidmaze' not in FLAGS.env_name or FLAGS.agent['discount'] == 0.995, (
        'Humanoid maze tasks require discount factor of 0.995.'
    )
    assert FLAGS.dataset_dir is not None, 'must provide dataset directory'

    ##=========== RANDOM SEED ===========##
    rng = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed); random.seed(FLAGS.seed)

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

    setup_wandb(project=FLAGS.wbproj, entity='moma1234', group=FLAGS.run_group, name=exp_name, id=wbid, resume=resume)
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
            title=f'{animal} run started!',
            text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}",
        )
    ##=========== END SET-UP ===========##

    ##=========== CREATE ENV, DATA ===========##
    total_steps = FLAGS.offline_steps + FLAGS.collection_steps + FLAGS.further_offline_steps
    config = FLAGS.agent
    further_training_start = FLAGS.offline_steps + FLAGS.collection_steps

    global_step_file = pathlib.Path(FLAGS.save_dir) / 'global_step'
    rbsize_file = pathlib.Path(FLAGS.save_dir) / 'rbsize'
    replaybuffer_pointer_file = pathlib.Path(FLAGS.save_dir) / 'replaybuffer_pointer'
    train_logger = None
    eval_logger = None
    restored_rbsize = None
    restored_replaybuffer_pointer = None

    if global_step_file.exists() and int(global_step_file.read_text().strip()) > 0:
        global_step = int(global_step_file.read_text().strip())
        print(f'Restoring from epoch {global_step}')
        if rbsize_file.exists():
            restored_rbsize = int(rbsize_file.read_text().strip())
        if replaybuffer_pointer_file.exists():
            restored_replaybuffer_pointer = int(replaybuffer_pointer_file.read_text().strip())

        np.random.seed(FLAGS.seed)
        dataset_path = pathlib.Path(FLAGS.save_dir) / f'data-{global_step}.npz'
        env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=str(dataset_path), use_oracle_reps=True)

        train_dataset_data = _load_npz_dict(dataset_path)
        val_dataset_data = _load_npz_dict(pathlib.Path(FLAGS.save_dir) / f'data-{global_step}-val.npz')

        train_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**train_dataset_data, freeze=False), config)
        val_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**val_dataset_data, freeze=False), config)
        example_batch = train_dataset.sample(1)

        agent = _create_agent(config['agent_name'], FLAGS.seed, example_batch, config)
        print(agent.config, file=sys.stderr)
        agent = restore_agent(agent, FLAGS.save_dir, global_step)

        collection_state = None
        if FLAGS.offline_steps <= global_step < further_training_start:
            collection_state_path = pathlib.Path(FLAGS.save_dir) / f'collection_state-{global_step}.pkl'
            if collection_state_path.exists():
                import pickle
                with open(collection_state_path, 'rb') as f:
                    collection_state = pickle.load(f)
                # Recreate env
                env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=str(dataset_path), use_oracle_reps=True)
                collection_state['env'] = env
                print(f'Loaded collection_state from {collection_state_path}')

        if global_step < FLAGS.offline_steps:
            train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
            eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
            first_time = time.time()
            last_time = time.time()
        elif further_training_start <= global_step:
            train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
            eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))
            first_time = time.time()
            last_time = time.time()
    else:
        global_step = 0
        global_step_file.write_text(str(global_step))

        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
        dataset_idx = 0

        splits = FLAGS.env_name.split('-')
        env_name = '-'.join(splits[:-3]) + '-v0'
        env, train_dataset_data, val_dataset_data = make_env_and_datasets(
            env_name,
            dataset_path=datasets[dataset_idx],
            use_oracle_reps=True,
        )
        print(env.task_infos, file=sys.stderr)

        train_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**train_dataset_data, freeze=False), config)
        val_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**val_dataset_data, freeze=False), config)
        example_batch = train_dataset.sample(1)

        agent = _create_agent(config['agent_name'], FLAGS.seed, example_batch, config)
        print(agent.config, file=sys.stderr)

        collection_state = None

    ##=========== END CREATE ENV, DATA ===========##

    ##=========== MAIN LOOP ===========##
    with tqdm.tqdm(total=total_steps, initial=global_step) as pbar:
        while global_step < total_steps:
            if global_step < FLAGS.offline_steps:
                if global_step == 0:
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

            if FLAGS.offline_steps <= global_step < further_training_start:
                if collection_state is None:
                    if restored_rbsize is not None:
                        rbsize = restored_rbsize
                    else:
                        train_dataset_size = train_dataset.size
                        rbsize = train_dataset.size + FLAGS.collection_steps
                    if not isinstance(train_dataset, ReplayBuffer):
                        train_dataset = ReplayBuffer.create_from_initial_dataset(dict(_dataset_mapping(train_dataset)), rbsize)

                    if restored_replaybuffer_pointer is not None:
                        current_size = min(restored_replaybuffer_pointer, rbsize)
                    else:
                        current_size = train_dataset_size
                    train_dataset.pointer = current_size
                    train_dataset.size = current_size


                    env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx], use_oracle_reps=True)

                    ob, reset_info = env.reset()
                    del reset_info
                    goal = _get_task_goal(env)

                    print(goal)

                    done = False

                    stats = get_statistics_class(FLAGS.env_name)(env=env)
                    data_to_plot = {}

                    dataset_for_plot = _dataset_mapping(train_dataset)
                    floored = np.floor(dataset_for_plot['oracle_reps'])
                    for i in range(len(floored)):
                        rounded = (floored[i][0], floored[i][1])
                        if rounded in data_to_plot:
                            data_to_plot[rounded] += 1
                        else:
                            data_to_plot[rounded] = 1

                    fig_name = plot_heatmap(data_to_plot, save_dir=FLAGS.save_dir)
                    wandb.log({'data_collection/data_viz': wandb.Image(fig_name)}, step=global_step)
                    print(f'Plotted data to {fig_name}')
                    os.remove(fig_name)
                    
                    new_data_to_plot = {k: 0 for k in data_to_plot}
                    collection_state = dict(
                        ob=ob,
                        goal=goal,
                        done=done,
                        stats=stats,
                        data_to_plot=data_to_plot,
                        new_data_to_plot=new_data_to_plot,
                        env=env
                    )
                else:
                    ob = collection_state['ob']
                    goal = collection_state['goal']
                    done = collection_state['done']
                    stats = collection_state['stats']
                    data_to_plot = collection_state['data_to_plot']
                    new_data_to_plot = collection_state['new_data_to_plot']
                    env = collection_state['env']

                curr_rng, rng = jax.random.split(rng)
                action = agent.sample_actions(
                    observations=ob,
                    goals=goal,
                    seed=curr_rng,
                )
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                tran = dict(
                    observations=ob,
                    actions=action,
                    terminals=float(done),
                    next_observations=next_ob,
                    qpos=info['qpos'],
                    qvel=info['qvel'],
                    oracle_reps=to_oracle_rep(obs=ob[None], env=env)[0],
                    # masks=1.0 - terminated,
                    # rewards=reward
                )

                train_dataset.add_transition(tran)
                stats.log_episode(tran['observations'], tran['actions'])
                
                rounded = (np.floor(ob[0]), np.floor(ob[1])) # TODO: make this generalizable
                if rounded in data_to_plot:
                    data_to_plot[rounded] += 1
                    new_data_to_plot[rounded] += 1
                else:
                    data_to_plot[rounded] = 1
                    new_data_to_plot[rounded] = 1

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
                    collection_state=collection_state,
                )

                next_collection_ob = next_ob
                next_goal = goal

                if done:
                    next_collection_ob, reset_info = env.reset()
                    del reset_info
                    next_goal = _get_task_goal(env)
                    print(goal)
                    if terminated:
                        wandb.log({'data_collection/goal_reach': 1.0}, step=global_step)
                    else:
                        wandb.log({'data_collection/goal_reach': 0.0}, step=global_step)

                collection_state['ob'] = next_collection_ob
                collection_state['goal'] = next_goal
                collection_state['done'] = done

                if global_step % FLAGS.log_interval == 0:
                    for k, v in stats.get_statistics().items():
                        wandb.log({f'data_collection/{k}': v}, step=global_step)

                if global_step % FLAGS.data_plot_interval == 0:
                    fig_name = plot_heatmap(data_to_plot, save_dir=FLAGS.save_dir)
                    # TODO: plot heatmap
                    wandb.log({'data_collection/data_viz': wandb.Image(fig_name)}, step=global_step)
                    print(f'Plotted data to {fig_name}')
                    os.remove(fig_name)

                    fig_name = plot_heatmap(new_data_to_plot, save_dir=FLAGS.save_dir)
                    # TODO: plot heatmap
                    wandb.log({'data_collection/new_data_viz': wandb.Image(fig_name)}, step=global_step)
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

            if further_training_start <= global_step:

                if global_step == further_training_start:
                    fql_config = FLAGS.fql_agent
                    train_dataset['terminals'][train_dataset.size - 1] = 1.0
                    
                    # train_dataset = _wrap_goal_conditioned_dataset(train_dataset, fql_config)
                    datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
                    dataset_idx = 0

                    # env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx], use_oracle_reps=True)
                    print(f'new replay buffer size: {train_dataset.size}')
                    from flax.core import unfreeze
                    train_dataset = unfreeze(train_dataset)
                    relabel_dataset(FLAGS.env_name, env, train_dataset)
                    train_dataset = Dataset.create(**train_dataset, freeze=False)

                    example_batch = train_dataset.sample(1)
                    agent = _create_agent(fql_config['agent_name'], FLAGS.seed, example_batch, fql_config)

                    print(agent.config, file=sys.stderr)

                    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
                    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))
                    first_time = time.time()
                    last_time = time.time()
                    print(f'Beginning further training for {FLAGS.further_offline_steps} steps', file=sys.stderr)

                agent, global_step, last_time = _train_step(
                    agent=agent,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    config=fql_config,
                    env=env,
                    global_step=global_step,
                    pbar=pbar,
                    first_time=first_time,
                    last_time=last_time,
                    train_logger=train_logger,
                    eval_logger=eval_logger,
                    save_dir=FLAGS.save_dir,
                    global_step_file=global_step_file,
                    prefix='further'
                )

                if global_step == total_steps:
                    train_logger.close()
                    eval_logger.close()
                    if FLAGS.cleanup:
                        _cleanup_checkpoints(FLAGS.save_dir)

            run_metrics = {'global_step': global_step}
            train_dataset_size = train_dataset.size
            if train_dataset_size is not None:
                run_metrics['data/train_dataset_size'] = train_dataset_size
            wandb.log(run_metrics, step=global_step)


if __name__ == '__main__':
    app.run(main)
