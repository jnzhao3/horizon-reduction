import json
import os
import pathlib
import random
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
import wandb.util
from absl import app, flags
from flax.core import unfreeze
from ml_collections import config_flags

from agents import agents
from ogbench.relabel_utils import relabel_dataset
from utils.datasets import CGCDataset, Dataset, GCDataset, ReplayBuffer
from utils.evaluation import evaluate, evaluate_gcfql
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_animal, get_exp_name, get_flag_dict, setup_wandb
from utils.plot_utils import plot_heatmap
from utils.statistics import get_statistics_class
from wrappers.datafuncs_utils import make_env_and_datasets, to_oracle_reps

FLAGS = flags.FLAGS

flags.DEFINE_string('wbproj', 'aorl2', 'Weights & Biases project name.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('wandb_alerts', True, 'Enable Weights & Biases alerts.')

flags.DEFINE_string('env_name', 'humanoidmaze-large-navigate-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory. Kept for parity with triangle_inequality.py.')

config_flags.DEFINE_config_file('agent', 'agents/dqc.py', lock_config=False)
config_flags.DEFINE_config_file('fql_agent', 'agents/fql.py', lock_config=False)
flags.DEFINE_string('save_dir', '../../scratch', 'Save directory.')

flags.DEFINE_string(
    'restore_dir',
    '../../scratch/aorl2/2026-04-08-00/2026-04-08-00.b7bf8a914965d2ce2cdfd7704faa38b5fee704b874bc391d21e3b9137701759c/',
    'Directory containing the restored DQC checkpoint and dataset snapshots.',
)
flags.DEFINE_integer('restore_epoch', 1000000, 'Checkpoint step to restore.')

flags.DEFINE_integer('proposer_steps', 0, 'Number of finetuning steps for the ungoal-conditioned DQC proposer.')
flags.DEFINE_integer('collection_steps', 1000000, 'Number of online collection steps.')
flags.DEFINE_integer('further_offline_steps', 1000000, 'Number of FQL training steps after relabeling.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')
flags.DEFINE_integer('data_plot_interval', 100000, 'Data plotting interval.')
flags.DEFINE_bool('cleanup', False, 'If true, delete saved data and weight checkpoints at run end.')

flags.DEFINE_integer('steps_toward_sg', 200, 'Number of environment steps allocated to each subgoal before resampling.')
flags.DEFINE_float('subgoal_reached_distance', 0.25, 'Distance threshold to consider a subgoal reached.')
flags.DEFINE_integer('num_subgoal_candidates', 128, 'Number of candidate subgoals to sample from the proposer.')

flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')


def _dataset_mapping(dataset):
    if hasattr(dataset, 'dataset'):
        return dataset.dataset
    if hasattr(dataset, '_dict'):
        return dataset._dict
    return dataset


def _load_npz_dict(path):
    file = np.load(path)
    return {k: file[k][...] for k in file.files}


def _cleanup_checkpoints(save_dir):
    removed = []
    for pattern in ('params_*.pkl', 'data-*.npz'):
        for path in pathlib.Path(save_dir).glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    print(f'Cleanup enabled. Removed {len(removed)} checkpoint files.')


def _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step):
    save_agent(agent, save_dir, global_step)
    np.savez(os.path.join(save_dir, f'data-{global_step}.npz'), **_dataset_mapping(train_dataset))
    np.savez(os.path.join(save_dir, f'data-{global_step}-val.npz'), **_dataset_mapping(val_dataset))


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
    if config['dataset_class'] == 'CGCDataset':
        return CGCDataset(dataset, config)
    return GCDataset(dataset, config)


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
        if prefix:
            eval_info = {f'{prefix}{k}': v for k, v in eval_info.items()}
        return eval_info

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
            use_oracle_rep=True,
        )
        for metric_name, metric_value in eval_info.items():
            eval_metrics[f'{prefix}evaluation/{task_name}_{metric_name}'] = metric_value
            overall_metrics[metric_name].append(metric_value)

    for metric_name, values in overall_metrics.items():
        eval_metrics[f'{prefix}evaluation/overall_{metric_name}'] = np.mean(values)

    return eval_metrics


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
    prefix='',
    batch_transform=None,
):
    batch = train_dataset.sample(config['batch_size'])
    if batch_transform is not None:
        batch = batch_transform(batch)
    agent, update_info = agent.update(batch)
    global_step += 1
    pbar.update(1)

    if global_step % FLAGS.log_interval == 0:
        train_metrics = {f'{prefix}training/{k}': v for k, v in update_info.items()}
        val_batch = val_dataset.sample(config['batch_size'])
        if batch_transform is not None:
            val_batch = batch_transform(val_batch)
        _, val_info = agent.total_loss(val_batch, grad_params=None)
        train_metrics.update({f'{prefix}validation/{k}': v for k, v in val_info.items()})
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
        _save_agent_and_datasets(agent, train_dataset, val_dataset, save_dir, global_step)

    return agent, global_step, last_time


def _zero_goal_conditioning(batch):
    batch = dict(batch)
    if 'high_value_goals' in batch:
        batch['high_value_goals'] = np.zeros_like(batch['high_value_goals'])
    if 'high_actor_goals' in batch:
        batch['high_actor_goals'] = np.zeros_like(batch['high_actor_goals'])
    if 'actor_goals' in batch:
        batch['actor_goals'] = np.zeros_like(batch['actor_goals'])
    if 'value_goals' in batch:
        batch['value_goals'] = np.zeros_like(batch['value_goals'])
    return batch


def _sample_ungc_subgoal(agent, observation, rng, env):
    observation = np.asarray(observation)
    observations = np.repeat(observation[None], FLAGS.num_subgoal_candidates, axis=0)
    # DQC's proposer expects a goal input. For the ungoal-conditioned variant,
    # we zero that channel out and use the learned flow prior directly.
    dummy_goals = np.zeros((FLAGS.num_subgoal_candidates, agent.config['goal_dim']), dtype=np.float32)
    subgoals = np.asarray(agent.propose_goals(observations, dummy_goals, rng))

    dists = np.asarray(agent.compute_dynamical_distance(observations, subgoals, env))
    best_idx = int(np.argmax(dists))
    return subgoals[best_idx], dists, subgoals


def _plot_data_if_needed(data_to_plot, key, global_step):
    fig_name = plot_heatmap(data_to_plot, save_dir=FLAGS.save_dir)
    wandb.log({key: wandb.Image(fig_name)}, step=global_step)
    os.remove(fig_name)


def main(_):
    assert FLAGS.restore_dir is not None, 'must provide restore_dir'

    rng = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

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

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    if FLAGS.wandb_alerts:
        animal = get_animal()
        print(f'\n{animal}\n', exp_name)
        print('\n\n', info, file=sys.stderr)
        print('\n\npython', ' '.join(sys.argv), '\n', file=sys.stderr)
        wandb.run.alert(
            title=f'{animal} run started!',
            text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}",
        )

    restore_dir = pathlib.Path(FLAGS.restore_dir)
    train_dataset_path = restore_dir / f'data-{FLAGS.restore_epoch}.npz'
    val_dataset_path = restore_dir / f'data-{FLAGS.restore_epoch}-val.npz'
    assert train_dataset_path.exists(), f'missing restored dataset {train_dataset_path}'
    assert val_dataset_path.exists(), f'missing restored val dataset {val_dataset_path}'

    env, _, _ = make_env_and_datasets(FLAGS.env_name, dataset_path=str(train_dataset_path), use_oracle_reps=True)

    train_dataset_data = _load_npz_dict(train_dataset_path)
    val_dataset_data = _load_npz_dict(val_dataset_path)

    dqc_config = FLAGS.agent
    train_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**train_dataset_data, freeze=False), dqc_config)
    val_dataset = _wrap_goal_conditioned_dataset(Dataset.create(**val_dataset_data, freeze=False), dqc_config)
    example_batch = train_dataset.sample(1)

    dqc_agent = _create_agent(dqc_config['agent_name'], FLAGS.seed, example_batch, dqc_config)
    dqc_agent = restore_agent(dqc_agent, FLAGS.restore_dir, FLAGS.restore_epoch)
    print(dqc_agent.config, file=sys.stderr)

    proposer_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_proposer.csv'))
    proposer_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_proposer.csv'))
    collection_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'collection.csv'))
    fql_train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
    fql_eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))

    proposer_global_step = 0
    collection_global_step = FLAGS.proposer_steps
    fql_global_step = FLAGS.proposer_steps + FLAGS.collection_steps
    total_steps = FLAGS.proposer_steps + FLAGS.collection_steps + FLAGS.further_offline_steps
    first_time = time.time()
    last_time = first_time

    with tqdm.tqdm(total=total_steps) as pbar:
        if FLAGS.proposer_steps > 0:
            print(f'Finetuning DQC goal proposer for {FLAGS.proposer_steps} steps', file=sys.stderr)
            while proposer_global_step < FLAGS.proposer_steps:
                dqc_agent, proposer_global_step, last_time = _train_step(
                    agent=dqc_agent,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    config=dqc_config,
                    env=env,
                    global_step=proposer_global_step,
                    pbar=pbar,
                    first_time=first_time,
                    last_time=last_time,
                    train_logger=proposer_train_logger,
                    eval_logger=proposer_eval_logger,
                    save_dir=FLAGS.save_dir,
                    prefix='proposer/',
                    batch_transform=_zero_goal_conditioning,
                )

        dataset_for_collection = _dataset_mapping(train_dataset)
        if not isinstance(dataset_for_collection, ReplayBuffer):
            rbsize = dataset_for_collection.size + FLAGS.collection_steps
            replay_buffer = ReplayBuffer.create_from_initial_dataset(dict(dataset_for_collection), rbsize)
        else:
            replay_buffer = dataset_for_collection

        ob, reset_info = env.reset()
        del reset_info
        curr_rng, rng = jax.random.split(rng)
        subgoal, dists, _ = _sample_ungc_subgoal(dqc_agent, ob, curr_rng, env)
        subgoal_steps = 0
        stats = get_statistics_class(FLAGS.env_name)(env=env)
        done = False

        data_to_plot = {}
        new_data_to_plot = {}
        selected_subgoals = {}
        total_subgoals = 1
        reached_subgoals = 0

        floored = np.floor(np.asarray(replay_buffer['oracle_reps']))
        for i in range(min(replay_buffer.size, len(floored))):
            rounded = tuple(floored[i].tolist())
            data_to_plot[rounded] = data_to_plot.get(rounded, 0) + 1
            new_data_to_plot.setdefault(rounded, 0)

        rounded_subgoal = tuple(np.floor(subgoal).tolist())
        selected_subgoals[rounded_subgoal] = selected_subgoals.get(rounded_subgoal, 0) + 1

        if data_to_plot:
            _plot_data_if_needed(data_to_plot, 'data_collection/data_viz', collection_global_step)

        print(f'Collecting {FLAGS.collection_steps} online steps with DQC', file=sys.stderr)
        while collection_global_step < FLAGS.proposer_steps + FLAGS.collection_steps:
            curr_rng, rng = jax.random.split(rng)
            action = dqc_agent.sample_actions(observations=ob, goals=subgoal, seed=curr_rng)
            action = np.asarray(np.clip(action, -1, 1))

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            tran = dict(
                observations=ob,
                actions=action,
                terminals=float(done),
                next_observations=next_ob,
                qpos=info['qpos'],
                qvel=info['qvel'],
                oracle_reps=to_oracle_reps(obs=np.asarray(ob)[None], env=env)[0],
            )
            replay_buffer.add_transition(tran)
            stats.log_episode(tran['observations'], tran['actions'])

            rounded = tuple(np.floor(np.asarray(tran['oracle_reps'])).tolist())
            data_to_plot[rounded] = data_to_plot.get(rounded, 0) + 1
            new_data_to_plot[rounded] = new_data_to_plot.get(rounded, 0) + 1

            collection_global_step += 1
            pbar.update(1)

            next_collection_ob = next_ob
            subgoal_steps += 1
            next_rep = np.asarray(to_oracle_reps(np.asarray(next_ob)[None], env))[0]
            subgoal_done = np.linalg.norm(next_rep - np.asarray(subgoal)) <= FLAGS.subgoal_reached_distance
            subgoal_timed_out = subgoal_steps >= FLAGS.steps_toward_sg

            if done:
                next_collection_ob, reset_info = env.reset()
                del reset_info
                curr_rng, rng = jax.random.split(rng)
                subgoal, dists, _ = _sample_ungc_subgoal(dqc_agent, next_collection_ob, curr_rng, env)
                total_subgoals += 1
                subgoal_steps = 0
                wandb.log({'data_collection/goal_reach': 1.0 if terminated else 0.0}, step=collection_global_step)
            elif subgoal_done or subgoal_timed_out:
                if subgoal_done:
                    reached_subgoals += 1
                curr_rng, rng = jax.random.split(rng)
                subgoal, dists, _ = _sample_ungc_subgoal(dqc_agent, next_ob, curr_rng, env)
                rounded_subgoal = tuple(np.floor(subgoal).tolist())
                selected_subgoals[rounded_subgoal] = selected_subgoals.get(rounded_subgoal, 0) + 1
                total_subgoals += 1
                subgoal_steps = 0

            ob = next_collection_ob

            if collection_global_step % FLAGS.log_interval == 0:
                metrics = {f'data_collection/{k}': v for k, v in stats.get_statistics().items()}
                metrics['data_collection/subgoal_reach_fraction'] = reached_subgoals / max(total_subgoals, 1)
                metrics['data_collection/subgoal_distance_mean'] = float(np.mean(dists))
                metrics['data_collection/subgoal_distance_max'] = float(np.max(dists))
                wandb.log(metrics, step=collection_global_step)
                collection_logger.log(metrics, step=collection_global_step)

            if collection_global_step % FLAGS.data_plot_interval == 0:
                if data_to_plot:
                    _plot_data_if_needed(data_to_plot, 'data_collection/data_viz', collection_global_step)
                if new_data_to_plot:
                    _plot_data_if_needed(new_data_to_plot, 'data_collection/new_data_viz', collection_global_step)
                if selected_subgoals:
                    _plot_data_if_needed(selected_subgoals, 'data_collection/selected_subgoals_viz', collection_global_step)

            if collection_global_step % FLAGS.save_interval == 0:
                _save_agent_and_datasets(dqc_agent, replay_buffer, val_dataset, FLAGS.save_dir, collection_global_step)

        if replay_buffer.size > 0:
            replay_buffer['terminals'][replay_buffer.size - 1] = 1.0

        print('Relabeling collected replay buffer for FQL training', file=sys.stderr)
        relabeled_train_data = unfreeze(replay_buffer)
        relabel_dataset(FLAGS.env_name, env, relabeled_train_data)
        relabeled_train_dataset = Dataset.create(**relabeled_train_data, freeze=False)
        fql_val_dataset = Dataset.create(**val_dataset_data, freeze=False)

        fql_config = FLAGS.fql_agent
        fql_example_batch = relabeled_train_dataset.sample(1)
        fql_agent = _create_agent(fql_config['agent_name'], FLAGS.seed, fql_example_batch, fql_config)
        print(fql_agent.config, file=sys.stderr)

        print(f'Training FQL for {FLAGS.further_offline_steps} steps', file=sys.stderr)
        while fql_global_step < total_steps:
            fql_agent, fql_global_step, last_time = _train_step(
                agent=fql_agent,
                train_dataset=relabeled_train_dataset,
                val_dataset=fql_val_dataset,
                config=fql_config,
                env=env,
                global_step=fql_global_step,
                pbar=pbar,
                first_time=first_time,
                last_time=last_time,
                train_logger=fql_train_logger,
                eval_logger=fql_eval_logger,
                save_dir=FLAGS.save_dir,
                prefix='further/',
            )

    _save_agent_and_datasets(fql_agent, relabeled_train_dataset, fql_val_dataset, FLAGS.save_dir, total_steps)

    proposer_train_logger.close()
    proposer_eval_logger.close()
    collection_logger.close()
    fql_train_logger.close()
    fql_eval_logger.close()

    if FLAGS.cleanup:
        _cleanup_checkpoints(FLAGS.save_dir)


if __name__ == '__main__':
    app.run(main)
