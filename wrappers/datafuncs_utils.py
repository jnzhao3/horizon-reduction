import numpy as np
from utils.datasets import Dataset
import ogbench
from ogbench.relabel_utils import add_oracle_reps
import jax
import jax.numpy as jnp

def clip_dataset(train_dataset, N):
    if N > 0:
        new_train_dataset = {}
        if 'valids' in train_dataset:
            idxs = np.where(train_dataset['valids'] == 1)[0]
            idxs = idxs[N]
        else:
            idxs = N
        for k, v in train_dataset.items():
            # Ensure we have a writable host array
            if isinstance(v, np.ndarray):
                arr = v[:idxs].copy()                       # writable copy
            else:
                try:
                    # JAX DeviceArray, memmap, etc. -> force to NumPy writable
                    arr = np.array(v[:idxs], copy=True)
                except Exception:
                    # As a fallback (e.g., PyTorch tensor)
                    try:
                        arr = v[:idxs].clone().cpu().numpy()
                    except Exception:
                        arr = np.array(v[:idxs], copy=True)

            if k == "terminals":
                arr[idxs - 1] = 1  # cast to dtype automatically (bool->True, uint8->1)
            elif k == "valids":
                arr[idxs - 1] = 0

            new_train_dataset[k] = arr

        train_dataset = new_train_dataset
    return train_dataset

def make_env_and_datasets(dataset_name, dataset_path, dataset_only=False, cur_env=None, use_oracle_reps=False, env_only=False, terminate_at_goal=True, max_episode_steps=-1):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the environment (dataset).
        dataset_path: Path to the dataset file.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).

    Returns:
        A tuple of the environment (if `dataset_only` is False), training dataset, and validation dataset.
    """
    if env_only:
        if max_episode_steps > 0:
            env = ogbench.make_env_and_datasets(
                dataset_name, dataset_path=dataset_path, compact_dataset=False, env_only=env_only, cur_env=cur_env, add_info=True, terminate_at_goal=terminate_at_goal, max_episode_steps=max_episode_steps
            )
        return env
    if dataset_only:
        train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_path=dataset_path, compact_dataset=False, dataset_only=dataset_only, cur_env=cur_env, add_info=True, terminate_at_goal=terminate_at_goal
        )
    else:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, dataset_path=dataset_path, compact_dataset=False, dataset_only=dataset_only, cur_env=cur_env, add_info=True, terminate_at_goal=terminate_at_goal
        )

    if use_oracle_reps:
        add_oracle_reps(env.spec.id, env, train_dataset)
        add_oracle_reps(env.spec.id, env, val_dataset)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    # Clip dataset actions.
    eps = 1e-5
    train_dataset = train_dataset.copy(
        add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + eps, 1 - eps))
    )
    val_dataset = val_dataset.copy(add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + eps, 1 - eps)))

    if dataset_only:
        return train_dataset, val_dataset
    else:
        env.reset()
        return env, train_dataset, val_dataset