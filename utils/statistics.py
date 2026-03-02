from utils.plot_utils import get_block_i_pos_idxs
import math

def entropy(counts, total):
    ent = 0
    for c in counts:
        p = c / total
        ent += -p * math.log(p + 1e-10)
    return ent

class MazeStatistics:
    def __init__(self, **kwargs):
        # self.episode_lengths = []
        # self.episode_returns = []
        # self.success_rates = []
        self.cells_count = {} # cell identified by bottom-left corner
        self.cells_unit = 1
        self.num_transitions = 0

    def log_episode(self, observation, action, **kwargs):
        x_corner = int(observation[0] // self.cells_unit * self.cells_unit)
        y_corner = int(observation[1] // self.cells_unit * self.cells_unit)
        cell = (x_corner, y_corner)
        if cell not in self.cells_count:
            self.cells_count[cell] = 0
        self.cells_count[cell] += 1
        self.num_transitions += 1

    def get_statistics(self):
        assert self.num_transitions > 0, 'no transitions logged'
        num_cells_visited = len(self.cells_count)
        avg_visits_per_cell = self.num_transitions / num_cells_visited

        stats = {
            'num_cells_visited': num_cells_visited,
            'num_cells_visited_per_transition': num_cells_visited / self.num_transitions,
            'avg_visits_per_cell': avg_visits_per_cell,
            'max_visits_per_cell': max(self.cells_count.values()),
            'min_visits_per_cell': min(self.cells_count.values()),
            # 'most_visited_cell': max(self.cells_count, key=self.cells_count.get),
            # 'least_visited_cell': min(self.cells_count, key=self.cells_count.get),
            # 'cells_count': self.cells_count,
            'cells_entropy': entropy(self.cells_count.values(), self.num_transitions),
        }
        return stats

    def reset(self):
        self.cells_count = {}
        self.num_transitions = 0

class CubeStatistics:
    def __init__(self, env, **kwargs):
        self.num_cubes = env.task_infos[0]['init_xyzs'].shape[0]
        self.cells_count_per_cube = [{} for _ in range(self.num_cubes)] # cell identified by bottom-left corner
        self.cells_unit = 0.1
        self.num_transitions = 0

    def log_episode(self, observation, action, **kwargs):
        for i in range(self.num_cubes):
            x_idx, y_idx, z_idx = get_block_i_pos_idxs(i, self.num_cubes)
            x_corner = int(observation[x_idx] // self.cells_unit * self.cells_unit)
            y_corner = int(observation[y_idx] // self.cells_unit * self.cells_unit)
            cell = (x_corner, y_corner)
            if cell not in self.cells_count_per_cube[i]:
                self.cells_count_per_cube[i][cell] = 0
            self.cells_count_per_cube[i][cell] += 1
        self.num_transitions += 1

    def get_statistics(self):
        assert self.num_transitions > 0, 'no transitions logged'
        stats = {}
        for i in range(self.num_cubes):
            num_cells_visited = len(self.cells_count_per_cube[i])
            avg_visits_per_cell = self.num_transitions / num_cells_visited
            stats.update({
                f'cube_{i+1}_num_cells_visited': num_cells_visited,
                f'cube_{i+1}_num_cells_visited_per_transition': num_cells_visited / self.num_transitions,
                f'cube_{i+1}_avg_visits_per_cell': avg_visits_per_cell,
                f'cube_{i+1}_max_visits_per_cell': max(self.cells_count_per_cube[i].values()),
                f'cube_{i+1}_min_visits_per_cell': min(self.cells_count_per_cube[i].values()),
                # f'cube_{i+1}_most_visited_cell': max(self.cells_count_per_cube[i], key=self.cells_count_per_cube[i].get),
                # f'cube_{i+1}_least_visited_cell': min(self.cells_count_per_cube[i], key=self.cells_count_per_cube[i].get),
                # f'cube_{i+1}_cells_count': self.cells_count_per_cube[i],
                f'cube_{i+1}_cells_entropy': entropy(self.cells_count_per_cube[i].values(), self.num_transitions),
            })
        return stats

    def reset(self):
        self.cells_count_per_cube = [{} for _ in range(self.num_cubes)]
        self.num_transitions = 0
        

statistics = {
    'humanoidmaze-medium-navigate-v0' : MazeStatistics,
    'humanoidmaze-large-navigate-v0' : MazeStatistics,
    'humanoidmaze-medium-navigate-oraclerep-v0' : MazeStatistics,
    'humanoidmaze-large-navigate-oraclerep-v0' : MazeStatistics,
    'humanoidmaze-medium-v0' : MazeStatistics,
    'antmaze-medium-v0' : MazeStatistics,
    'antmaze-medium-navigate-oraclerep-v0': MazeStatistics,
    'antmaze-large-navigate-oraclerep-v0': MazeStatistics,
    'cube-triple-play-v0' : CubeStatistics,
    'cube-quadruple-play-v0' : CubeStatistics,
}