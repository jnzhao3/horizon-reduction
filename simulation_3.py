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
##============ END IMPORTS ==========##

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
    import ipdb; ipdb.set_trace()
    ##=========== END SET-UP ===========##

    ##=========== WANDB SET-UP ===========##
    if not args.debug:
        run = wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY,              # or your username / team name
            name=f"{NAME}_{task_start}_{task_end}_{waypoint}",    # optional run name
            group=NAME,
            config=dict(
                task_start=task_start, task_end=task_end, waypoint=waypoint, collection_steps=args.collection_steps, action_noise=args.action_noise, train_steps=args.train_steps, eval_interval=args.eval_interval
            ),
            mode="online",            # or "offline" if no internet
        )

    ##=========== CREATE REPLAY BUFFER ===========##
    original_size = train_dataset.size
    rbsize = original_size + args.collection_steps
    import ipdb; ipdb.set_trace()

    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset.dataset), rbsize)
    train_dataset.size = rbsize
    import ipdb; ipdb.set_trace() # CHECK POINTER, SIZE
    ##=========== END CREATE REPLAY BUFFER===========##

    ##=========== COLLECT DATA ===========##
    # TODO: calculate waypoint separately
    import pickle
    with open('cells.pkl', 'rb') as f:
        potential_goals = pickle.load(f)

    start_ij = env.unwrapped.xy_to_ij(waypoint)
    goal_xy = np.random.choice(potential_goals)
    goal_ij = env.unwrapped.xy_to_ij(goal_xy)
    import ipdb; ipdb.set_trace()
    ob, _ = env.reset(options=dict(
            task_info=dict(
                init_ij=start_ij,
                goal_ij=goal_ij
            )
        )
    )
    done = False
    counter = 0
    rng = jax.random.PRNGKey(seed)
    print(f'Collection {args.collection_steps} transitions now!')

    with tqdm.tqdm(total=args.collection_steps) as pbar:
        while counter < args.collection_steps:
            
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

            tran = dict(
                observations=ob,
                actions=action,
                terminals=float(done),
                next_observations=next_ob,
                qpos=info['qpos'],
                qvel=info['qvel'],
            )

            import ipdb; ipdb.set_trace()
            tran['oracle_reps'] = ob[:2]

            import ipdb; ipdb.set_trace()
            train_dataset.add_transition(tran)
            counter += 1; pbar.update(1)

            if done:
                goal_xy = np.random.choice(potential_goals)
                goal_ij = env.unwrapped.xy_to_ij(goal_xy)
                ob, _ = env.reset(options=dict(
                        task_info=dict(
                            init_ij=start_ij,
                            goal_ij=goal_ij
                        )
                    )
                )
                done = False

    ##=========== TRAIN ===========##
    print('training now!')

    env.task_infos = [{
                'task_name': f'{task_start} to {task_end}',
                'init_ij': env.unwrapped.xy_to_ij(task_start),
                'init_xy': task_start,
                'goal_ij': env.unwrapped.xy_to_ij(task_end),
                'goal_xy': task_end
            }]
    
    for step in tqdm(args.train_steps):

        batch = train_dataset.sample(batch_size)

        import ipdb; ipdb.set_trace()
        agent, update_info = agent.update(batch)
        
        ##=========== EVALUATION ===========##
        if step % args.eval_interval == 0:
            eval_metrics, all_trajs = env.evaluate_step(agent=agent, config=config, eval_episodes=5, return_trajs=True)

            ##=========== LOG AND PLOT WANDB ===========##
            if not args.debug:
                wandb.log(eval_metrics, step=step)

            import ipdb; ipdb.set_trace()

    ##=========== SAVE DATA AND CHECKPOINTS ===========##
    os.makedirs(f'~/scratch/{NAME}', exist_ok=True)
    save_agent(agent, f'~/scratch/{NAME}/{run.name}', step)
    np.savez(os.path.join(f'~/scratch/{NAME}', f'data-{step}.npz'), **train_dataset.dataset)

    if not args.debug:
        run.finish()

if __name__ == '__main__':
    ##=========== ARGUMENTS ===========##
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_start', type=str, help='description for option1')
    parser.add_argument('--task_end', type=str, help='description for option2')
    parser.add_argument('--waypoint', type=str, help='description for option3')
    parser.add_argument('--collection_steps', type=int, default=1000000, help='number of additional data collection steps to take')
    parser.add_argument('--action_noise', type=float, default=0)
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--eval_interval', type=int, default=100000)
    parser.add_argument('--debug', action='store_true')
    ##=========== END ARGUMENTS ===========##
    args = parser.parse_args()
    main(args)