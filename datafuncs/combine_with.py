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
class CombineWith:
    '''
    Given an original dataset, and specifications for new datasets to combine with,
    create a new dataset that combines them.
    '''

    # def create(original_dataset, config, agent_config, env, **kwargs):
    #     # return cls()
    #     example_transition = original_dataset.sample(1)
    #     example_transition = jax.tree_util.tree_map(lambda x: x[0], example_transition)
    #     rbsize = sum(config['train_data_sizes'])
    #     replay_buffer = ReplayBuffer.create(dict(example_transition), rbsize)

    #     if len(config['train_data_keys']) > 0:
    #         assert False, 'train_data_keys not supported yet'
    #     else:
    #         for path, size in zip(config['train_data_paths'], config['train_data_sizes']):
    #             import os
    #             path = os.path.expanduser(os.path.join(config['dataset_dir'], path))
    #             # new_data = Dataset.load(path)
    #             ob_dtype = np.uint8 if ('visual' in path or 'powderworld' in path) else np.float32
    #             action_dtype = np.int32 if 'powderworld' in path else np.float32

    #             new_data = load_dataset(
    #                 path,
    #                 ob_dtype=ob_dtype,
    #                 action_dtype=action_dtype,
    #                 compact_dataset=True,
    #                 add_info=True,
    #             )

    #             add_oracle_reps(env.spec.id, env, new_data)

    #             # dataset_class_dict = {
    #             #     'GCDataset': GCDataset,
    #             #     'HGCDataset': HGCDataset,
    #             # }
    #             # dataset_class = dataset_class_dict[config['dataset_class']]
    #             new_data = clip_dataset(new_data, size)
    #             dataset_class = type(original_dataset)
    #             new_data = dataset_class(Dataset.create(**new_data), agent_config)
    #             # val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    #             import ipdb; ipdb.set_trace()
    #             replay_buffer.combine_with(new_data)

    #     return replay_buffer
    
    def create(original_dataset, config, agent_config, env, **kwargs):

        original_dataset_dict = original_dataset.dataset.unfreeze()
        rbsize = original_dataset_dict['observations'].shape[0] + sum(config['train_data_sizes'])

        if len(config['train_data_keys']) > 0:
            assert False, 'train_data_keys not supported yet'
        else:
            import datafuncs.data as data
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

        original_dataset_dict = jax.tree_util.tree_map(lambda x: jnp.array(x), original_dataset_dict)
        original_dataset = Dataset.create(**original_dataset_dict, freeze=False)
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
            method_name='combine_with',
            # combine_ratio=0.5,  # ratio of original data to keep when combining datasets
            # new_data_path='',  # path to new dataset to combine with
            # seed=0,
            train_data_keys=(),
            train_data_paths="gcfql_main_maze3_5",
            train_data_sizes=(),
            dataset_dir="../../scratch"
        )
    )
    return config