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

def make_ogbench_env_and_datasets(
        dataset_name,
        dataset_dir='~/.ogbench/data',
        dataset_path=None,
        dataset_size=None,
        compact_dataset=False,
        env_only=False,
        dataset_only=False,
        cur_env=None,
        add_info=False,
        **env_kwargs,
):
    """Make OGBench environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save the datasets.
        dataset_path: (Optional) Path to the dataset.
        dataset_size: (Optional) Size of the dataset.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        dataset_only: Whether to return only the dataset.
        cur_env: Current environment (only used when `dataset_only` is True).
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.
    """
    # Make environment.
    splits = dataset_name.split('-')
    dataset_add_info = add_info
    env = cur_env
    if 'singletask' in splits:
        # Single-task environment.
        pos = splits.index('singletask')
        env_name = '-'.join(splits[: pos - 1] + splits[pos:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)
        dataset_name = '-'.join(splits[:pos] + splits[-1:])  # Remove the words 'singletask' and 'task\d' (if exists).
        dataset_add_info = True
    elif 'oraclerep' in splits:
        # Environment with oracle goal representations.
        env_name = '-'.join(splits[:-3] + splits[-1:])  # Remove the dataset type and the word 'oraclerep'.
        if not dataset_only:
            env = gymnasium.make(env_name, use_oracle_rep=True, **env_kwargs)
        dataset_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the word 'oraclerep'.
        dataset_add_info = True
    else:
        # Original, goal-conditioned environment.
        env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
        if not dataset_only:
            env = gymnasium.make(env_name, **env_kwargs)

    if env_only:
        return env

    # Load datasets.
    if dataset_path is None:
        dataset_dir = os.path.expanduser(dataset_dir)
        ogbench.download_datasets([dataset_name], dataset_dir)
        train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
        val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    else:
        train_dataset_path = dataset_path
        val_dataset_path = dataset_path.replace('.npz', '-val.npz')

    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
        dataset_size=dataset_size,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
        dataset_size=dataset_size,
    )

    if 'singletask' in splits:
        # Add reward information to the datasets.
        from ogbench.relabel_utils import relabel_dataset
        relabel_dataset(env_name, env, train_dataset)
        relabel_dataset(env_name, env, val_dataset)

    if 'oraclerep' in splits:
        # Add oracle goal representations to the datasets.
        from ogbench.relabel_utils import add_oracle_reps
        add_oracle_reps(env_name, env, train_dataset)
        add_oracle_reps(env_name, env, val_dataset)

    if not add_info:
        # Remove information keys.
        for k in ['qpos', 'qvel', 'button_states']:
            if k in train_dataset:
                del train_dataset[k]
            if k in val_dataset:
                del val_dataset[k]

    if dataset_only:
        return train_dataset, val_dataset
    else:
        return env, train_dataset, val_dataset


def to_oracle_reps(obs, env=None, env_name=None):
    """Add oracle goal representations to the dataset.

    Args:
        env_name: Name of the environment.
        env: Environment.
        dataset: Dataset dictionary.
    """
    assert env or env_name, "Must provide environment name or environment."
    obs = jnp.asarray(obs)
    if env_name is None:
        env_name = env.spec.id
    if 'maze' in env_name or 'soccer' in env_name:
        # Locomotion environments.
        qpos_xy_start_idx = 0
        qpos_ball_start_idx = 15

        if 'maze' in env_name:
            oracle_reps = obs[:, qpos_xy_start_idx : qpos_xy_start_idx + 2]
            return oracle_reps
        else:
            oracle_reps = obs[:, qpos_ball_start_idx : qpos_ball_start_idx + 2]
            return oracle_reps
    elif 'cube' in env_name or 'scene' in env_name or 'puzzle' in env_name:
        # Manipulation environments.
        qpos_obj_start_idx = 14
        qpos_cube_length = 7
        xyz_center = jnp.array([0.425, 0.0, 0.0], dtype=obs.dtype)
        xyz_scaler = 10.0
        drawer_scaler = 18.0
        window_scaler = 15.0

        if 'cube' in env_name:
            num_cubes = env.unwrapped._num_cubes

            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    obs[
                        :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
                    ]
                )
            cube_xyzs = jnp.stack(cube_xyzs_list, axis=1)
            oracle_reps = ((cube_xyzs - xyz_center) * xyz_scaler).reshape(-1, num_cubes * 3)
            return oracle_reps
        # elif 'scene' in env_name:
        #     num_cubes = env.unwrapped._num_cubes
        #     num_buttons = env.unwrapped._num_buttons
        #     qpos_drawer_idx = qpos_obj_start_idx + num_cubes * qpos_cube_length + num_buttons
        #     qpos_window_idx = qpos_drawer_idx + 1

        #     cube_xyzs_list = []
        #     for i in range(num_cubes):
        #         cube_xyzs_list.append(
        #             obs[
        #                 :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
        #             ]
        #         )
        #     cube_xyzs = np.stack(cube_xyzs_list, axis=1)
        #     cube_reps = ((cube_xyzs - xyz_center) * xyz_scaler).reshape(-1, num_cubes * 3)
        #     button_reps = dataset['button_states'].copy()
        #     drawer_reps = dataset['qpos'][:, [qpos_drawer_idx]] * drawer_scaler
        #     window_reps = dataset['qpos'][:, [qpos_window_idx]] * window_scaler
        #     oracle_reps = np.concatenate([cube_reps, button_reps, drawer_reps, window_reps], axis=-1)
        # elif 'puzzle' in env_name:
        #     oracle_reps = dataset['button_states'].copy()
    else:
        raise ValueError(f'Unsupported environment: {env_name}')
