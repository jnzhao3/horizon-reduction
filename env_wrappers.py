import gymnasium
import numpy as np
from collections import defaultdict
import tqdm
from utils.log_utils import get_wandb_video
from utils.evaluation import evaluate_gcfql, evaluate_custom_gcfql

class MazeEnvWrapper(gymnasium.Wrapper):

    '''
    Wrapper to handle deployment efficient setting.
    '''

    def __init__(self, env, seed=0):
        super().__init__(env)
        self.seed = seed
        self.all_cells = []
        self.vertex_cells = []
        maze_map = env.unwrapped.maze_map
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 0:
                    self.all_cells.append((i, j))

                    # Exclude hallway cells.
                    if (
                        maze_map[i - 1, j] == 0
                        and maze_map[i + 1, j] == 0
                        and maze_map[i, j - 1] == 1
                        and maze_map[i, j + 1] == 1
                    ):
                        continue
                    if (
                        maze_map[i, j - 1] == 0
                        and maze_map[i, j + 1] == 0
                        and maze_map[i - 1, j] == 1
                        and maze_map[i + 1, j] == 1
                    ):
                        continue

                    self.vertex_cells.append((i, j))

        np.random.seed(0)
        random_idxes = np.random.choice(len(self.vertex_cells), size=5, replace=False)

        task_info = []
        for task_i in range(1, 6):
            # i, j = self.vertex_cells[np.random.randint(len(self.vertex_cells))]
            i, j = self.vertex_cells[random_idxes[task_i - 1]]
            task_info.append({
                'task_name': f'custom_task{task_i}',
                # 'init_ij': start['init_ij'],
                'goal_ij': (i, j),
                'goal_xy': self.unwrapped.ij_to_xy((i, j)),
            })

        remaining_vertex_cells = [self.vertex_cells[i] for i in range(len(self.vertex_cells)) if i not in random_idxes]

        np.random.seed(seed)
        start = {'init_ij': remaining_vertex_cells[np.random.randint(len(remaining_vertex_cells))]}
        start['init_xy'] = self.unwrapped.ij_to_xy(start['init_ij'])
        self.start_ij = start['init_ij']
        self.start_xy = start['init_xy']

        for info in task_info:
            info['init_ij'] = start['init_ij']
            info['init_xy'] = start['init_xy']

        self.task_infos = task_info

        self.all_cells = np.array([env.unwrapped.ij_to_xy(ij) for ij in self.all_cells])
        self.vertex_cells = np.array([env.unwrapped.ij_to_xy(ij) for ij in self.vertex_cells])

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)
    
    def step(self, action):
        return super().step(action)
    
    def evaluate_step(self, 
                        agent, 
                        config,
                        env_name='Maze',
                        eval_episodes=10,
                        video_episodes=0,
                        video_frame_skip=3,
                        eval_temperature=0,
                        eval_gaussian=None,
                        ):
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        # if task_info is not None:
        #     task_infos = task_info
        # else:
        #     task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = len(self.task_infos)
        for task_id in tqdm.trange(1, num_tasks + 1):
            task_name = self.task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = evaluate_custom_gcfql(
                agent=agent,
                env=self,
                env_name=env_name,
                goal_conditioned=True,
                # task_id=task_id,
                init_ij=self.task_infos[task_id - 1]['init_ij'],
                goal_ij=self.task_infos[task_id - 1]['goal_ij'],
                config=config,
                num_eval_episodes=eval_episodes,
                num_video_episodes=video_episodes,
                video_frame_skip=video_frame_skip,
                eval_temperature=eval_temperature,
                eval_gaussian=eval_gaussian,
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
            )
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)
        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        if video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=5)
                eval_metrics['video'] = video

        return eval_metrics