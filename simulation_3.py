##=========== IMPORTS ===========##
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
from utils.flax_utils import restore_agent, save_agent
from agents import agents
from utils.datasets import GCDataset, Dataset, ReplayBuffer
import pathlib
from wrappers.datafuncs_utils import make_env_and_datasets
from env_wrappers import MazeEnvWrapper
import jax
from tqdm import tqdm
import pickle
##============ END IMPORTS ==========##

##=========== ARGUMENTS ===========##
parser = argparse.ArgumentParser()
parser.add_argument('--task_start', type=str, help='description for option1')
parser.add_argument('--task_end', type=str, help='description for option2')
parser.add_argument('--waypoint', type=str, help='description for option3')
parser.add_argument('--collection_steps', type=int, default=1000000, help='number of additional data collection steps to take')
parser.add_argument('--action_noise', type=float, default=0)
parser.add_argument('--train_steps', type=int, default=1000000)
parser.add_argument('--eval_interval', type=int, default=10000)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num_eval_episodes', type=int, default=5)
parser.add_argument('--experiment', type=str, default='')
##=========== END ARGUMENTS ===========##

##=========== CONSTANTS ===========##
NAME = 'simulation_3'
PROJECT_NAME = 'aorl'
ENTITY = 'jnzhao3'
PATH = '../../scratch/aorl/e2e_maze_8_2/e2e_maze_8_2.a5406616b3951230fafa4ca948138750f5881270649f6138153e795abde02629'
RUN_NAME = 'e2e_maze_8_2.a5406616b3951230fafa4ca948138750f5881270649f6138153e795abde02629'
DATA_STEP = 100000
AGENT_STEP = 1000000

##=========== END CONSTANTS ===========##

def main(args):

    task_start = eval(args.task_start)
    task_end = eval(args.task_end)
    waypoint = eval(args.waypoint)

    ID_NAME = f"{NAME}_{task_start}_{task_end}_{waypoint}"
    ID_NAME = ID_NAME.replace('(', '')
    ID_NAME = ID_NAME.replace(')', '')
    DIR = f'../../scratch/{NAME}/{ID_NAME}'
    print(DIR, file=sys.stderr)
    os.makedirs(DIR, exist_ok=True)
    
    ##=========== SET-UP ===========##
    api = wandb.Api()
    run = api.runs(f'{ENTITY}/{PROJECT_NAME}', filters={'display_name': f'{RUN_NAME}'})[0]

    config = run.config['agent']
    agent_class = agents[config['agent_name']]

    dataset_path = pathlib.Path(PATH) / f'data-{DATA_STEP}.npz'
    env, train_dataset, val_dataset = make_env_and_datasets(run.config['env_name'], dataset_path=str(dataset_path), use_oracle_reps=True)
    dataset_class = GCDataset
    train_dataset = dataset_class(Dataset.create(**train_dataset, freeze=False), config)

    example_batch = train_dataset.sample(1)
    seed = run.config['seed']
    batch_size = run.config['agent']['batch_size']
    agent = agent_class.create(
        seed,
        example_batch,
        config
    )

    agent = restore_agent(agent, PATH, AGENT_STEP)

    env = MazeEnvWrapper(env)
    all_cells = env.all_cells
    ##=========== END SET-UP ===========##

    ##=========== WANDB SET-UP ===========##
    if not args.debug:
        run = wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY,              # or your username / team name
            name=ID_NAME,    # optional run name
            group=NAME,
            config=dict(
                task_start=task_start, task_end=task_end, waypoint=waypoint, collection_steps=args.collection_steps, action_noise=args.action_noise, train_steps=args.train_steps, eval_interval=args.eval_interval, experiment=args.experiment
            ),
            mode="online",            # or "offline" if no internet
            resume="never"
        )
        wandb.run.alert(title=f"{ID_NAME} run started!", text=f"{ID_NAME}\n\n{'python ' + ' '.join(sys.argv)}\n\n{run.id}")

        if "SLURM_JOB_ID" in os.environ:
            print(os.environ["SLURM_JOB_ID"], file=sys.stderr) # for debugging purposes, check that this is the correct value
            run.config.update({
                'job': os.environ["SLURM_JOB_ID"],
            }, allow_val_change=True)

    ##=========== CREATE REPLAY BUFFER ===========##
    original_size = train_dataset.size
    rbsize = original_size + args.collection_steps

    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset.dataset), rbsize)
    train_dataset.size = rbsize
    ##=========== END CREATE REPLAY BUFFER===========##

    ##=========== COLLECT DATA ===========##
    # TODO: calculate waypoint separately
    with open('cells.pkl', 'rb') as f:
        potential_goals = pickle.load(f)

    start_ij = env.unwrapped.xy_to_ij(waypoint) # IMPORTANT: this is waypoint, not task_start
    # goal_xy = potential_goals[np.random.choice(range(len(potential_goals)))]
    # goal_ij = env.unwrapped.xy_to_ij(goal_xy)
    # ob, _ = env.reset(options=dict(
    #         task_info=dict(
    #             init_ij=start_ij,
    #             goal_ij=goal_ij
    #         )
    #     )
    # )
    # done = False
    done = True; terminated = False; truncated = False
    counter = 0
    rng = jax.random.PRNGKey(seed)
    print(f'Collection {args.collection_steps} transitions now!', file=sys.stderr)

    with tqdm(total=args.collection_steps) as pbar:
        while counter < args.collection_steps:

            if done:
                print(f'terminated: {terminated}, truncated: {truncated}', file=sys.stderr)
                goal_xy = potential_goals[np.random.choice(range(len(potential_goals)))]
                goal_ij = env.unwrapped.xy_to_ij(goal_xy)
                ob, _ = env.reset(options=dict(
                        task_info=dict(
                            init_ij=start_ij,
                            goal_ij=goal_ij
                        )
                    )
                )
                done = False
            
            curr_rng, rng = jax.random.split(rng)
            action = agent.sample_actions(
                observations=ob,
                goals=goal_xy,
                seed=curr_rng,
            )

            action = action + np.random.normal(0, args.action_noise, size=action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # make sure the last transition is terminal!
            if counter == (args.collection_steps - 1):
                done = True

            tran = dict(
                observations=ob,
                actions=action,
                terminals=float(done),
                next_observations=next_ob,
                qpos=info['qpos'],
                qvel=info['qvel'],
            )

            tran['oracle_reps'] = ob[:2]

            train_dataset.add_transition(tran)
            counter += 1; pbar.update(1)
            ob = next_ob

    ##=========== TRAIN ===========##
    print('training now!', file=sys.stderr)

    train_dataset = GCDataset(Dataset.create(**train_dataset), config)

    np.savez(os.path.join(DIR, f'data-{args.collection_steps}.npz'), **train_dataset.dataset)

    ##=========== PLOT THINGS ===========##
    all_cells = env.all_cells
    fig_name = f'{DIR}/plot.png'
    plt.figure(figsize=(6, 6))
    plt.scatter(x=all_cells[:, 0], y=all_cells[:, 1], s=10, c='gray')
    plt.scatter(x=[task_start[0]], y=[task_start[1]], s=50, c='red')
    plt.scatter(x=[task_end[0]], y=[task_end[1]], s=50, c='green')
    plt.scatter(x=[waypoint[0]], y=[waypoint[1]], s=50, c='blue', marker='*')
    plt.savefig(fig_name)
    if not args.debug:
        wandb.log({"data_collection/plot": wandb.Image(fig_name)}, step=counter)
        os.remove(fig_name)

    fig_name = f'{DIR}/collected_data.png'
    plt.figure(figsize=(6, 6))
    plt.scatter(x=all_cells[:, 0], y=all_cells[:, 1], s=10, c='gray')

    new_data = train_dataset.dataset['observations'][original_size :, :2]
    plt.scatter(x=new_data[:, 0], y=new_data[:, 1], s=10, c='orange', alpha=0.002)
    plt.scatter(x=[task_start[0]], y=[task_start[1]], s=50, c='red')
    plt.scatter(x=[task_end[0]], y=[task_end[1]], s=50, c='green')
    plt.scatter(x=[waypoint[0]], y=[waypoint[1]], s=50, c='blue', marker='*')
    plt.savefig(fig_name)
    if not args.debug:
        wandb.log({"data_collection/plot": wandb.Image(fig_name)}, step=counter)
        os.remove(fig_name)
    ##=========== END PLOT THINGS ===========##

    env.task_infos = [{
                'task_name': f'{task_start} to {task_end}',
                'init_ij': env.unwrapped.xy_to_ij(task_start),
                'init_xy': task_start,
                'goal_ij': env.unwrapped.xy_to_ij(task_end),
                'goal_xy': task_end
            }]
    
    for step in tqdm(range(args.train_steps)):

        batch = train_dataset.sample(batch_size)
        agent, update_info = agent.update(batch)
        
        ##=========== EVALUATION ===========##
        if step % args.eval_interval == 0:
            eval_metrics, all_trajs = env.evaluate_step(agent=agent, config=config, eval_episodes=args.num_eval_episodes, return_trajs=True)
            with open(f'{DIR}/all_trajs-{step}.pkl', 'wb') as f:
                pickle.dump(all_trajs, f)

            save_agent(agent, DIR, step)

            ##=========== LOG AND PLOT WANDB ===========##
            if not args.debug:
                wandb.log(eval_metrics, step=counter + step) # to prevent overriding previous logs

    ##=========== END SAVE DATA AND CHECKPOINTS ===========##


    if not args.debug:
        run.finish()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)