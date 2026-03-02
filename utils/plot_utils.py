from matplotlib.pylab import f
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
import os

def bfs(init_ij, goal_ij, all_cells):
    s = init_ij
    all_cells_dict = {cell : [False, None] for cell in all_cells}
    q = []
    while s != goal_ij:
        all_cells_dict[s][0] = True
        neighbors = [(s[0] - 1, s[1]),(s[0] + 1, s[1]),(s[0], s[1] - 1),(s[0], s[1] + 1)]
        for n in neighbors:
            if n in all_cells_dict and not all_cells_dict[n][0]:
                all_cells_dict[n][1] = s
                q.append(n)
        s = q.pop(0)

    optimal_path = []
    p = goal_ij
    while p != init_ij:
        optimal_path.append(p)
        p = all_cells_dict[p][1]
    optimal_path.append(p)
    optimal_path = optimal_path[::-1]
    return optimal_path

def plot_data(points_config, save_dir=None):
    plt.clf()
    for k, points_set in points_config.items():
        plt.scatter(**points_set, label=k)

    plt.legend()

    current_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    points_name = f"points-{current_date}.png"
    if save_dir:
        points_name = os.path.join(save_dir, points_name)
        plt.savefig(points_name)
        return points_name
    else:
        plt.show()
        return None


def plot_points(all_cells, optimal_path, points, save_dir=None, other_points=[]):
    x_all_cells = np.array(all_cells)[:, 0]
    y_all_cells = np.array(all_cells)[:, 1]
    x_optimal_path = np.array(optimal_path)[:, 0]
    y_optimal_path = np.array(optimal_path)[:, 1]
    x_spots = np.array(points)[:, 0]
    y_spots = np.array(points)[:, 1]

    plt.clf()
    plt.scatter(x_all_cells, y_all_cells, alpha=1, color="gray")
    plt.scatter(x_optimal_path, y_optimal_path, alpha=1, color="blue")
    plt.scatter(x_spots, y_spots, s=10, c=range(len(x_spots)), cmap="viridis")

    if other_points:
        plt.scatter(np.array(other_points)[:, 0], np.array(other_points)[:, 1], s=40, color="green")
    # plt.xlim(-5, 40)
    # plt.ylim(-5, 30)
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    points_name = f"points-{current_date}.png"
    if save_dir:
        points_name = os.path.join(save_dir, points_name)
    plt.savefig(points_name)
    return points_name

def plot_replay(all_cells, optimal_path, replay_buffer, train_dataset, num_added_trans, save_dir=None):
    x_all_cells = np.array(all_cells)[:, 0]
    y_all_cells = np.array(all_cells)[:, 1]
    x_optimal_path = np.array(optimal_path)[:, 0]
    y_optimal_path = np.array(optimal_path)[:, 1]

    # len_train = len(train_dataset["observations"])
    len_train = train_dataset.size
    replay_spots = replay_buffer["observations"][len_train: len_train + num_added_trans]
    x_ob_viz = replay_spots[:, 0]
    y_ob_viz = replay_spots[:, 1]

    plt.clf()
    plt.scatter(x_all_cells, y_all_cells, alpha=1, color="gray")
    plt.scatter(x_optimal_path, y_optimal_path, alpha=1, color="blue")
    plt.scatter(x_ob_viz, y_ob_viz, alpha=0.1, c=range(len(x_ob_viz)), cmap="viridis")
    # plt.xlim(-5, 40)
    # plt.ylim(-5, 30)
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    replay_name = f"replay-{current_date}.png"
    if save_dir:
        replay_name = os.path.join(save_dir, replay_name)
    plt.savefig(replay_name)
    return replay_name

def plot_replay_zero(all_cells, optimal_path, replay_buffer, num_added_trans, num_to_plot=None, save_dir=None):
    x_all_cells = np.array(all_cells)[:, 0]
    y_all_cells = np.array(all_cells)[:, 1]
    x_optimal_path = np.array(optimal_path)[:, 0]
    y_optimal_path = np.array(optimal_path)[:, 1]

    # len_train = len(train_dataset["observations"])
    # len_train = train_dataset.size
    if num_to_plot is None:
        start_idx = 0
    else:
        start_idx = num_added_trans - num_to_plot
    replay_spots = replay_buffer["observations"][start_idx: num_added_trans]
    x_ob_viz = replay_spots[:, 0]
    y_ob_viz = replay_spots[:, 1]

    plt.clf()
    plt.scatter(x_all_cells, y_all_cells, alpha=1, color="gray")
    plt.scatter(x_optimal_path, y_optimal_path, alpha=1, color="blue")
    plt.scatter(x_ob_viz, y_ob_viz, alpha=0.1, c=range(len(x_ob_viz)), cmap="viridis")
    # plt.xlim(-5, 40)
    # plt.ylim(-5, 30)
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    replay_name = f"replay-{current_date}.png"
    if save_dir:
        replay_name = os.path.join(save_dir, replay_name)
    plt.savefig(replay_name)
    return replay_name

def get_closest(xy, train_dataset):
    """
    Get the closest point in the training dataset to the given xy coordinate.
    """
    distances = np.linalg.norm(train_dataset["observations"][:, :2] - xy, axis=1)
    closest_index = np.argmin(distances)
    return train_dataset["observations"][closest_index]

def calculate_all_cells(env):
    all_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

    return all_cells

import time
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cube_task(buffer, env, plot_goals=False, num_to_plot=None, fig_name='', save_dir=''):
    total_start = time.time()

    # plt.clf()
    # fig = plt.figure()
    # ax = plt.axes()
    # print(f"[{time.time() - total_start:.3f}s] Initialized figure and axes.")

    observations = buffer['observations']
    num_cubes = env.task_infos[0]['init_xyzs'].shape[0]
    end_idx = buffer.size - 1
    # print(f"[{time.time() - total_start:.3f}s] Loaded observations and metadata.")

    if num_to_plot is None:
        start_idx = 0
    else:
        start_idx = end_idx + 1 - num_to_plot

    # x = observations[start_idx : end_idx + 1, 12]
    # y = observations[start_idx : end_idx + 1, 13]
    # z = observations[start_idx : end_idx + 1, 14]
    # print(f"[{time.time() - total_start:.3f}s] Extracted end-effector trajectory.")

#     cube_xyzs = []
#     qpos_obj_start_idx = 14
#     qpos_cube_length = 7
#     for i in range(num_cubes):
#         cube_xyzs.append(
#             buffer['qpos'][
#                 :, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3
#             ]
#         )
#     cube_xyzs = np.array(cube_xyzs)
#     print(f"[{time.time() - total_start:.3f}s] Extracted cube trajectories.")
# # 
#     ax.scatter(x, y, c=range(start_idx, end_idx + 1), cmap='viridis')
#     for i in range(num_cubes):
#         ax.scatter(cube_xyzs[i, :, 0], cube_xyzs[i, :, 1], c='red')
#     # print(f"[{time.time() - total_start:.3f}s] Plotted trajectories.")

#     if fig_name:
#         fig_name = os.path.join(save_dir, fig_name)
#     else:
#         fig_name = time.strftime("%Y-%m-%d_%H-%M-%S.png", time.localtime())
#         fig_name = os.path.join(save_dir, fig_name)

#     plt.savefig(fig_name)
    # print(f"[{time.time() - total_start:.3f}s] Saved figure to {fig_name}")
    plots = {}
    for i in range(num_cubes):
        plt.clf()
        fig = plt.figure()
        ax = plt.axes()

        x_idx, y_idx, z_idx = get_block_i_pos_idxs(i, num_cubes)
        x = observations[start_idx : end_idx + 1, x_idx]
        y = observations[start_idx : end_idx + 1, y_idx]

        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        
        ax.scatter(x, y, c=range(start_idx, end_idx + 1), cmap='viridis')
        init_xyz = env.task_infos[env.cur_task_id - 1]['init_xyzs'][i] - xyz_center
        init_xyz = init_xyz * xyz_scaler
        goal_xyz = env.task_infos[env.cur_task_id - 1]['goal_xyzs'][i] - xyz_center
        goal_xyz = goal_xyz * xyz_scaler
        ax.scatter(init_xyz[0], init_xyz[1], c='green', s=100, label='init')
        if plot_goals:
            ax.scatter(goal_xyz[0], goal_xyz[1], c='red', s=100, label='goal')
        ax.set_title(f'Cube {i}')
        ax.legend()

        if fig_name:
            fig_name_i = fig_name.replace('.png', f'_cube_{i}.png')
            fig_name_i = os.path.join(save_dir, fig_name_i)
        else:
            fig_name_i = time.strftime(f"%Y-%m-%d_%H-%M-%S_cube_{i}.png", time.localtime())
            fig_name_i = os.path.join(save_dir, fig_name_i)

        plt.savefig(fig_name_i)
        plots[f'cube_{i}'] = fig_name_i

    return plots

def get_block_i_pos_idxs(i, num_blocks):
    BASE = 19
    STRIDE = 9

    start_i = BASE + STRIDE * i
    block_i_pos = [start_i, start_i + 1, start_i + 2]
    return block_i_pos
    