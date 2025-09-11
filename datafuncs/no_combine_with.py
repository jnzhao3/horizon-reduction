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

@struct.dataclass
class CombineWith:

    def create(cls, original_dataset, config, env, **kwargs):
        # return cls()
        example_transition = original_dataset.sample(1)
        example_transition = jax.tree_map(lambda x: x[0], example_transition)
        rbsize = sum(config['train_data_sizes'])
        replay_buffer = ReplayBuffer.create(dict(example_transition), rbsize)

        if len(config['train_data_keys']) > 0:
            assert False, 'train_data_keys not supported yet'
        else:
            for path, size in zip(config['train_data_paths'], config['train_data_sizes']):
                import os
                path = os.path.expanduser(os.path.join(config['dataset_dir'], path))
                # new_data = Dataset.load(path)
                ob_dtype = np.uint8 if ('visual' in path or 'powderworld' in path) else np.float32
                action_dtype = np.int32 if 'powderworld' in path else np.float32

                new_data = load_dataset(
                    path,
                    ob_dtype=ob_dtype,
                    action_dtype=action_dtype,
                    compact_dataset=False,
                    add_info=True,
                )
                new_data = clip_dataset(new_data, size)

                add_oracle_reps(env.sped.id, env, new_data)
                import ipdb; ipdb.set_trace()
                replay_buffer.combine_with(Dataset.create(**new_data))

        return replay_buffer

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            method_name='combine_with',
            # combine_ratio=0.5,  # ratio of original data to keep when combining datasets
            # new_data_path='',  # path to new dataset to combine with
            # seed=0,
            train_data_keys=(),
            train_data_paths=("a", "b"),
            train_data_sizes=(),
            dataset_dir="../../scratch"
        )
    )
    return config