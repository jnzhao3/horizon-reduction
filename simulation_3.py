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
##============ END IMPORTS ==========##

##=========== ARGUMENTS ===========##
parser = argparse.ArgumentParser()
parser.add_argument('--task_start', help='description for option1')
parser.add_argument('--task_end', help='description for option2')
parser.add_argument('--waypoint', help='description for option3')
parser.add_argument('--collection_steps', default=1000000, help='number of additional data collection steps to take')
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

def main(task_start, task_end, waypoint, collection_steps, **kwargs):

    import ipdb; ipdb.set_trace()
    
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
    agent = agent_class.create(
        run.config['seed'],
        example_batch,
        config
    )

    agent = restore_agent(agent, PATH, AGENT_STEP)

    env = MazeEnvWrapper(env)
    all_cells = env.all_cells
    ##=========== END SET-UP ===========##

    ##=========== CREATE REPLAY BUFFER ===========##
    original_size = train_dataset.size
    rbsize = original_size + collection_steps
    import ipdb; ipdb.set_trace()

    train_dataset = ReplayBuffer.create_from_initial_dataset(dict(train_dataset.dataset), rbsize)
    import ipdb; ipdb.set_trace() # CHECK POINTER, SIZE
    ##=========== END CREATE REPLAY BUFFER===========##

    ##=========== DETERMINE DATA COLLECTION POINT ===========##
    # TODO: calculate waypoint separately

    ob, _ = env.reset(options=dict(
        task_info=
    )

    done = False

    ##=========== COLLECT DATA ===========##

    ##=========== TRAIN ===========##

    ##=========== EVALUATION ===========##

    ##=========== LOG AND PLOT WANDB ===========##

    ##=========== SAVE DATA AND CHECKPOINTS ===========##


    return None

if __name__ == '__main__':
    args = parser.parse_args()
    main(**args)