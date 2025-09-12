from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import ml_collections 
from typing import Any
from ogbench.relabel_utils import add_oracle_reps
from utils.datasets import Dataset, ReplayBuffer
from datafuncs.datafuncs_utils import clip_dataset
from ogbench.relabel_utils import add_oracle_reps
from ogbench import load_dataset
import os
from utils.datasets import GCDataset, HGCDataset

@struct.dataclass
class RandomSteps:
    '''
    Given an original dataset, and specifications for new datasets to combine with,
    create a new dataset that combines them.
    '''
    
    def create(original_dataset, config, agent_config, env, **kwargs):
        '''
        Should return an expanded dataset that combines the original dataset with new datasets.
        '''

        # original_dataset_dict = original_dataset.dataset.unfreeze()
        # rbsize = original_dataset_dict['observations'].shape[0] + sum(config['train_data_sizes'])
        rbsize = original_dataset.size + config['collection_steps']
        replay_buffer = ReplayBuffer.create_from_initial_dataset(original_dataset, rbsize)

        i
            train_data_paths = eval(f"data.{config['train_data_paths']}") 
            for path, size in zip(train_data_paths, config['train_data_sizes']):
                # load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False, add_info=False)

                path = os.path.expanduser(os.path.join(config['dataset_dir'], path))
                ob_dtype = np.uint8 if ('visual' in path or 'powderworld' in path) else np.float32
                action_dtype = np.int32 if 'powderworld' in path else np.float32

                new_data = load_dataset(
                    path,
                    ob_dtype=ob_dtype,
                    action_dtype=action_dtype,
                    compact_dataset=True,
                    add_info=True,
                )

                add_oracle_reps(env.spec.id, env, new_data)
                new_data = clip_dataset(new_data, size)

                original_dataset_dict = jax.tree_util.tree_map(
                    lambda x, y: np.concatenate([x, y], axis=0),
                    original_dataset_dict,
                    new_data,
                )

        original_dataset = Dataset.create(**original_dataset_dict)
        dataset_class_dict = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }
        dataset_class = dataset_class_dict[agent_config['dataset_class']]
        train_dataset = dataset_class(original_dataset, agent_config)

        return train_dataset

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='randomsteps',
            collection_steps=1000000,
        )
    )
    return config