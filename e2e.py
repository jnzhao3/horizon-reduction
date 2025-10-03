import glob
import json
import os, signal, pathlib
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
import wandb.util

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, ReplayBuffer
from utils.evaluation import evaluate_gcfql, evaluate_custom_gcfql
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb, get_animal

import numpy as np
from utils.datasets import Dataset
from wrappers.datafuncs_utils import clip_dataset, make_env_and_datasets
from utils.plot_utils import plot_data
import gymnasium
from utils.samplers import to_oracle_rep
from wrappers import wrappers
from utils.plot_utils import plot_data
from env_wrappers import MazeEnvWrapper
from utils.statistics import statistics

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
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

##=========== TRAINING HYPERPARAMETERS ===========##
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 100000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')
flags.DEFINE_integer('collection_steps', 1000000, 'Number of data collection steps.')
flags.DEFINE_integer('data_plot_interval', 100000, 'Data plotting interval.')

##=========== EVALUATION HYPERPARAMETERS ===========##
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

##=========== DATA COLLECTION FLAGS ===========##
config_flags.DEFINE_config_file('wrapper', None, lock_config=False)
##=========== END FLAGS ===========##

# PREEMPTED = {"flag": False, "initial_trained": False, "data_collected": False, "further_trained": False}
PREEMPTED = {"flag": False}

def handle_preempt(signum, frame):
    PREEMPTED["flag"] = True

    print("Received preemption signal. Will save and exit after current epoch.", file=sys.stderr)

for sig in (signal.SIGUSR1, signal.SIGTERM):
    signal.signal(sig, handle_preempt)

def checkpoint_and_exit(agent, train_dataset, save_dir, global_step,
                        train_logger=None, eval_logger=None, *,
                        reason="preempt"):
    # atomic write for step
    tmp = pathlib.Path(save_dir) / "global_step.tmp"
    final = pathlib.Path(save_dir) / "global_step"
    tmp.write_text(str(global_step))
    tmp.replace(final)

    # agent & data
    save_agent(agent, save_dir, global_step)
    # make sure you’re saving the right dict structure:
    if hasattr(train_dataset, "dataset"):
        np.savez(os.path.join(save_dir, f"data-{global_step}.npz"), **train_dataset.dataset)
    else:
        np.savez(os.path.join(save_dir, f"data-{global_step}.npz"), **train_dataset)

    # close logs, finish wandb
    if train_logger: train_logger.close()
    if eval_logger:  eval_logger.close()
    try:
        wandb.alert(title="Preempted", text=f"Checkpointed at step {global_step} ({reason})")
    except Exception:
        pass
    try:
        wandb.finish()
    except Exception:
        pass
    # exit fast and clean
    raise SystemExit(0)

##=========== MAIN SCRIPT ===========##
def main(_):
    ##=========== ASSERTIONS ===========##
    assert 'humanoidmaze' not in FLAGS.env_name or FLAGS.agent['discount'] == 0.995, "Humanoid maze tasks require discount factor of 0.995."
    assert FLAGS.dataset_dir is not None, 'must provide dataset directory'
    assert FLAGS.agent['actor_type'] == 'best-of-n', "evaluation only implemented for best-of-n actors"
    ##=========== END ASSERTIONS ===========##

    # Set up logger.
    exp_name, info = get_exp_name(seed=FLAGS.seed, config=FLAGS)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wbproj, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    wandb_id_file = pathlib.Path(FLAGS.save_dir) / "wandb_id"
    if wandb_id_file.exists():
        wbid = wandb_id_file.read_text().strip()
        resume = 'allow'
    else:
        wbid = wandb.util.generate_id()
        wandb_id_file.write_text(wbid)
        resume = None
        
    setup_wandb(project='aorl', group=FLAGS.run_group, name=exp_name, id=wbid, resume=resume)
    wandb.run.config.update({'info': info}, allow_val_change=True)
    print(f"Created new run {wandb.run.name} with ID {wandb.run.id}")

    ##=========== LOG MESSAGES TO ERR AND SLACK ===========##
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    if FLAGS.wandb_alerts:
        animal = get_animal()
        print(f"\n{animal}\n", exp_name)
        print("\n\n", info, file=sys.stderr)
        print("\n\npython", " ".join(sys.argv), "\n", file=sys.stderr)
        wandb.run.alert(title=f"{animal} e2e run started!", text=f"{exp_name}\n\n{'python ' + ' '.join(sys.argv)}\n\n{info}")

    ##=========== RESTORE FROM RUN ===========##

    ##=========== TOTAL STEPS ===========##
    total_steps = 2 * FLAGS.offline_steps + FLAGS.collection_steps
    ##======================##============##

    global_step_file = pathlib.Path(FLAGS.save_dir) / "global_step"
    if global_step_file.exists() and int(global_step_file.read_text().strip()) > 0:
        global_step = int(global_step_file.read_text().strip())
        print(f"Restoring from epoch {global_step}")

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

        dataset_path = pathlib.Path(FLAGS.save_dir) / f"data-{global_step}.npz"
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=dataset_path, use_oracle_reps=True)
        env = MazeEnvWrapper(env, seed=FLAGS.seed)

        random.seed(FLAGS.seed); np.random.seed(FLAGS.seed)

        # start_ij = choose_start_ij(env)['init_ij']
        # task_info = create_task_infos(env, start_ij=start_ij)
        print(f"Evaluating on {len(env.task_infos)} tasks with start_ij {env.start_ij}")

        dataset_class_dict = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }

        dataset_class = dataset_class_dict[FLAGS.agent['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset, freeze=False), FLAGS.agent)
        val_dataset = dataset_class(Dataset.create(**val_dataset, freeze=False), FLAGS.agent)
        example_batch = train_dataset.sample(1)

        agent_class = agents[FLAGS.agent['agent_name']]
        agent = agent_class.create(
            FLAGS.seed,
            example_batch,
            FLAGS.agent,
        )
        print(agent.config, file=sys.stderr)
        agent = restore_agent(agent, FLAGS.save_dir, global_step)

        data_collection_env = None
    else:
        global_step = 0
        global_step_file.write_text(str(global_step))

        # Set up environment and datasets.
        config = FLAGS.agent
        datasets = [file for file in sorted(glob.glob(f'{FLAGS.dataset_dir}/*.npz')) if '-val.npz' not in file]
        dataset_idx = 0
        env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, dataset_path=datasets[dataset_idx], use_oracle_reps=True)
        env = MazeEnvWrapper(env, seed=FLAGS.seed)

        N = int(FLAGS.train_data_size)
        train_dataset = clip_dataset(train_dataset, N)

        # Initialize agent.
        random.seed(FLAGS.seed); np.random.seed(FLAGS.seed)
        print(f"Evaluating on {len(env.task_infos)} tasks with start_ij {env.start_ij}")

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

        if FLAGS.restore_path is not None:
            agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        data_collection_env = None

    ##=========== MAIN LOOP ===========##
    with tqdm.tqdm(total=total_steps) as pbar:
        while global_step < total_steps:

            ##=========== INITIAL TRAINING ===========##
            if global_step < FLAGS.offline_steps:
                if global_step == 0:
                    task_info_to_plot = {
                        'start_xy' : {'x': [env.start_xy[0]], 'y': [env.start_xy[1]], 's': 50,'c': 'red'},
                        'all_cells' : {'x': env.all_cells[:, 0], 'y': env.all_cells[:, 1], 's': 1, 'c': 'lightgrey'},
                    }
                    for t in env.task_infos:
                        task_info_to_plot[t['task_name']] = {
                            'x': t['goal_xy'][0], 'y': t['goal_xy'][1], 's': 50,
                            'c': random.choice(['blue', 'green', 'orange', 'purple', 'brown']),
                            'marker': random.choice(['*', 'X', 'P', 'D', 'v']),
                        }
                    fig_name = plot_data(task_info_to_plot, save_dir=FLAGS.save_dir)
                    wandb.log({"data_collection/task_info_viz": wandb.Image(fig_name)})
                    print(f"Plotted task info to {fig_name}"); os.remove(fig_name)

                    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
                    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
                    first_time = time.time(); last_time = time.time()
                    print(f"Beginning training for {FLAGS.offline_steps} steps", file=sys.stderr)

                ##=========== TRAINING ===========##
                batch = train_dataset.sample(config['batch_size'])
                agent, update_info = agent.update(batch)
                global_step += 1; pbar.update(1)
                if PREEMPTED["flag"]:
                    checkpoint_and_exit(agent, train_dataset, FLAGS.save_dir, global_step,
                                        train_logger if 'train_logger' in locals() else None,
                                        eval_logger   if 'eval_logger'   in locals() else None,
                                        reason="signal")

                ##=========== END TRAINING ===========##

                # Log metrics.
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

                # Evaluate agent.
                if FLAGS.eval_interval != 0 and (global_step == 1 or global_step % FLAGS.eval_interval == 0):
                    eval_metrics = env.evaluate_step(agent, 
                                                    config, 
                                                    env_name=FLAGS.env_name,
                                                    eval_episodes=FLAGS.eval_episodes,
                                                    video_episodes=FLAGS.video_episodes,
                                                    video_frame_skip=FLAGS.video_frame_skip,
                                                    eval_temperature=FLAGS.eval_temperature,
                                                    eval_gaussian=FLAGS.eval_gaussian,
                                                    )
                    wandb.log(eval_metrics, step=global_step)
                    eval_logger.log(eval_metrics, step=global_step)

                # Save agent.
                if global_step % FLAGS.save_interval == 0:
                    save_agent(agent, FLAGS.save_dir, global_step)
                    np.savez(os.path.join(FLAGS.save_dir, f"data-{global_step}.npz"), **train_dataset.dataset)
                    global_step_file.write_text(str(global_step))
            
                if global_step == FLAGS.offline_steps:
                    train_logger.close(); eval_logger.close()

            ##=========== DATA COLLECTION STEPS ===========##
            if FLAGS.offline_steps <= global_step and global_step < FLAGS.offline_steps + FLAGS.collection_steps:
                
                ##=========== DATA COLLECTION ===========##
                if data_collection_env is None:
                    data_collection_env = make_env_and_datasets(
                        FLAGS.env_name, dataset_path='', env_only=True, 
                        use_oracle_reps=True,
                        terminate_at_goal=False,
                        max_episode_steps=FLAGS.wrapper.get('max_episode_steps', 2000),
                    )

                    wrapper = wrappers[FLAGS.wrapper['method_name']]

                    clipped_dataset = clip_dataset(train_dataset.dataset, FLAGS.train_data_size)
                    collection_agent = wrapper.create(
                        agent=agent, 
                        train_dataset=clipped_dataset, 
                        config=FLAGS.wrapper)

                    rbsize = FLAGS.train_data_size + FLAGS.collection_steps
                    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset.dataset), rbsize)
                    rng = jax.random.PRNGKey(FLAGS.seed)

                    collection_agent, pre_info = collection_agent.pre()
                    for k, v in pre_info.items():
                        wandb.log({f'data_collection/pre/{k}': v}, step=global_step)

                    goal_xy = collection_agent.curr_goal; goal_ij = data_collection_env.unwrapped.xy_to_ij(goal_xy)
                    ob, _ = data_collection_env.reset(options=dict(task_info=dict(init_ij=env.start_ij, goal_ij=goal_ij)))
                    done = False; episode_return = 0; episode_length = 0

                    all_cells = env.all_cells
                    vertex_cells = env.vertex_cells
                    data_to_plot = {
                        'all_cells' : {'x': all_cells[:, 0], 'y': all_cells[:, 1], 's': 1, 'c': 'lightgrey'},
                        'vertex_cells' : {'x': vertex_cells[:, 0], 'y': vertex_cells[:, 1], 's': 5, 'c': 'grey'},
                        'buffer' : {'x': [], 'y': [], 's': 1, 'c': [], 'cmap': 'plasma'},
                        'goals' : { 'x': [goal_xy[0]], 'y': [goal_xy[1]], 's': 50, 'c': [global_step], 'cmap': 'viridis', 'marker': '*' }
                    }

                    stats = statistics[FLAGS.env_name](env=data_collection_env)
                else:
                    ##=========== PRE ===========##
                    collection_agent, pre_info = collection_agent.pre()
                    for k, v in pre_info.items():
                        wandb.log({f'data_collection/pre/{k}': v}, step=global_step)

                ##=========== STEP ENVIRONMENT ===========##
                curr_rng, rng = jax.random.split(rng)
                action, action_info = collection_agent.sample_actions(
                    observations=ob,
                    goals=None,
                    seed=curr_rng,
                    pre_info=pre_info,
                )
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = data_collection_env.step(action)
                done = terminated or truncated
                episode_return += reward; episode_length += 1

                tran = dict(
                    observations=ob,
                    actions=action,
                    terminals=float(done),
                    next_observations=next_ob,
                    qpos=info['qpos'],
                    qvel=info['qvel'],
                    oracle_reps = to_oracle_rep(obs=ob[None], env=env)[0],
                )

                train_dataset.add_transition(tran)
                data_to_plot['buffer']['x'].append(tran['oracle_reps'][0])
                data_to_plot['buffer']['y'].append(tran['oracle_reps'][1])
                data_to_plot['buffer']['c'].append(global_step)
                stats.log_episode(tran['observations'], tran['actions'])
                ##=========== END STEP ENVIRONMENT ===========##

                ##=========== POST ===========##
                collection_agent, post_info = collection_agent.post(transition=tran)
                for k, v in post_info.items():
                    wandb.log({f'data_collection/post/{k}': v}, step=global_step)
                global_step += 1; pbar.update(1)
                if PREEMPTED["flag"]:
                    checkpoint_and_exit(agent, train_dataset, FLAGS.save_dir, global_step,
                                        train_logger if 'train_logger' in locals() else None,
                                        eval_logger   if 'eval_logger'   in locals() else None,
                                        reason="signal")
                ##=========== END POST ===========##

                if done:
                    goal_xy = collection_agent.curr_goal; goal_ij = data_collection_env.unwrapped.xy_to_ij(goal_xy)
                    ob, _ = data_collection_env.reset(options=dict(task_info=dict(init_ij=env.start_ij, goal_ij=goal_ij)))

                    data_to_plot['goals']['x'].append(goal_xy[0])
                    data_to_plot['goals']['y'].append(goal_xy[1])
                    data_to_plot['goals']['c'].append(global_step)

                else:
                    ob = next_ob

                if global_step & FLAGS.log_interval == 0:
                    for k, v in stats.get_statistics().items():
                        wandb.log({f"data_collection/{k}": v}, step=global_step)

                if global_step % FLAGS.data_plot_interval == 0:
                    fig_name = plot_data(
                        data_to_plot,
                        save_dir=FLAGS.save_dir,
                    )
                    wandb.log({"data_collection/data_viz": wandb.Image(fig_name)}, step=global_step)
                    print(f"Plotted data to {fig_name}"); os.remove(fig_name)

                if global_step % FLAGS.save_interval == 0:
                    save_agent(agent, FLAGS.save_dir, global_step)
                    np.savez(os.path.join(FLAGS.save_dir, f"data-{global_step}.npz"), **train_dataset)
                    global_step_file.write_text(str(global_step))

                if global_step == FLAGS.offline_steps + FLAGS.collection_steps:
                    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
                    print(f'new replay buffer size: {train_dataset.size}')

            ##=========== FURTHER TRAINING ===========##
            if FLAGS.offline_steps + FLAGS.collection_steps <= global_step:

                ##=========== FURTHER TRAINING ===========##
                if global_step == FLAGS.offline_steps + FLAGS.collection_steps:
                    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train_further.csv'))
                    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval_further.csv'))
                    first_time = time.time(); last_time = time.time()
                    print(f"Beginning further training for {FLAGS.offline_steps} steps", file=sys.stderr)

                ##=========== TRAINING ===========##
                batch = train_dataset.sample(config['batch_size'])
                agent, update_info = agent.update(batch)
                global_step += 1; pbar.update(1)
                if PREEMPTED["flag"]:
                    checkpoint_and_exit(agent, train_dataset, FLAGS.save_dir, global_step,
                                        train_logger if 'train_logger' in locals() else None,
                                        eval_logger   if 'eval_logger'   in locals() else None,
                                        reason="signal")
                ##=========== END TRAINING ===========##

                # Log metrics.
                if global_step % FLAGS.log_interval == 0:
                    train_metrics = {f'training/{k}': v for k, v in update_info.items()}

                    val_batch = val_dataset.sample(config['batch_size'])
                    # val_batch = to_jnp(val_batch)
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

                    train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                    train_metrics['time/total_time'] = time.time() - first_time
                    last_time = time.time()
                    wandb.log(train_metrics, step=global_step)
                    train_logger.log(train_metrics, step=global_step)

                # Evaluate agent.
                if FLAGS.eval_interval != 0 and (global_step == 1 or global_step % FLAGS.eval_interval == 0):
                    eval_metrics = env.evaluate_step(agent, 
                                                    config,
                                                    env_name=FLAGS.env_name,
                                                    eval_episodes=FLAGS.eval_episodes,
                                                    video_episodes=FLAGS.video_episodes,
                                                    video_frame_skip=FLAGS.video_frame_skip,
                                                    eval_temperature=FLAGS.eval_temperature,
                                                    eval_gaussian=FLAGS.eval_gaussian,
                                                    )

                    wandb.log(eval_metrics, step=global_step)
                    eval_logger.log(eval_metrics, step=global_step)

                # Save agent.
                if global_step % FLAGS.save_interval == 0:
                    save_agent(agent, FLAGS.save_dir, global_step)
                    np.savez(os.path.join(FLAGS.save_dir, f"data-{global_step}.npz"), **train_dataset.dataset)
                    global_step_file.write_text(str(global_step))

                ##=========== END MAIN LOOP ===========##
                if global_step == total_steps:
                    train_logger.close(); eval_logger.close()

if __name__ == '__main__':
    app.run(main)