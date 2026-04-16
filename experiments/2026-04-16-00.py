from argparse import ArgumentParser
from pathlib import Path

from generate import SbatchGenerator
from generate_local import LocalScriptGenerator


parser = ArgumentParser()
parser.add_argument('--gen', type=str, default='brc', help='where to run the script')
parser.add_argument('--num_jobs_per_gpu', type=int, default=1, help='the number of jobs to allocate per gpu')
parser.add_argument('--gpu_limit', type=int, default=100)
args = parser.parse_args()


run_group = '2026-04-16-00'
data_root = '../../scratch/data/'
output_dir = Path(__file__).resolve().parents[1] / 'sbatch'
output_dir.mkdir(parents=True, exist_ok=True)
run_file = 'fql_baseline.py'
priority = 'high'


env_name = 'humanoidmaze-navigate-giant'
dataset_dir = data_root + env_name
discounts = [0.995, 0.999]
seeds = [1001, 2002, 3003]


for debug in [True, False]:
    if args.gen == 'local':
        gen = LocalScriptGenerator(prefix=('MUJOCO_GL=egl', f'python {run_file}'))
    else:
        gen = SbatchGenerator(
            j=args.num_jobs_per_gpu,
            limit=args.gpu_limit,
            prefix=('MUJOCO_GL=egl', f'python {run_file}'),
            comment=run_group,
            priority=priority,
        )

    if debug:
        gen.add_common_prefix({
            'run_group': run_group + '_debug',
            'offline_steps': 50,
            'further_offline_steps': 50,
            'collection_steps': 50,
            'eval_episodes': 1,
            'video_episodes': 0,
            'eval_interval': 5,
            'save_interval': 25,
            'data_plot_interval': 25,
            'log_interval': 25,
            'cleanup': True,
        })
    else:
        gen.add_common_prefix({
            'run_group': run_group,
            'offline_steps': 1000000,
            'further_offline_steps': 1000000,
            'collection_steps': 1000000,
            'save_interval': 100000,
            'eval_interval': 50000,
            'cleanup': True,
        })

    run_count = 0
    for seed in seeds:
        if debug and seed != 1001:
            break
        for discount in discounts:
            gen.add_run({
                'seed': seed,
                'env_name': env_name,
                'dataset_dir': dataset_dir,
                'agent': 'agents/fql.py',
                'agent.best_of_n': 4,
                'agent.horizon_length': 25,
                'agent.action_chunking': False,
                'agent.num_qs': 10,
                'agent.q_agg': 'mean',
                'agent.discount': discount,
                'agent.batch_size': 256,
                'agent.alpha': 30.0,
            })
            run_count += 1

    if run_count == 0:
        raise ValueError(f'No runs generated for debug={debug}.')

    generated = gen.generate_str()
    script_strs = [generated] if isinstance(generated, str) else generated
    name_prefix = f'{run_group}_{args.gen}_debug' if debug else f'{run_group}_{args.gen}'
    multi_part = len(script_strs) > 1

    for i, script_str in enumerate(script_strs, start=1):
        part_suffix = f'_part{i}' if multi_part else ''
        with open(output_dir / f'{name_prefix}{part_suffix}.sh', 'w') as f:
            f.write(script_str)
