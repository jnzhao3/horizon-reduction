from argparse import ArgumentParser
from itertools import product
from pathlib import Path

from generate import SbatchGenerator
from generate_local import LocalScriptGenerator


parser = ArgumentParser()
parser.add_argument('--gen', type=str, default='brc', help='where to run the script')
parser.add_argument('--num_jobs_per_gpu', type=int, default=1, help='the number of jobs to allocate per gpu')
parser.add_argument('--gpu_limit', type=int, default=100)
args = parser.parse_args()


run_group = '2026-04-29-03'
output_dir = Path(__file__).resolve().parents[1] / 'sbatch'
output_dir.mkdir(parents=True, exist_ok=True)
run_file = '19_giant_restored.py'
priority = 'high'

RESTORE_PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'
DATASET_DIR = '../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/'

ENV_NAME = 'humanoidmaze-giant-navigate-v0'
CKPT_NUM = 1_000_000

REPLAY_BUFFER_PATHS = [
    '../../scratch/checkpoints/data_collection_giant/rb_path_1.npz',
    '../../scratch/checkpoints/data_collection_giant/rb_path_2.npz',
    '../../scratch/checkpoints/data_collection_giant/rb_path_3.npz',
]


def make_command(
    *,
    replay_buffer_path,
    fql_train_steps,
    fql_n_step,
    fql_discount,
    fql_alpha,
    fql_best_of_n,
    seed,
    debug,
):
    flag_args = {
        'restore_path': RESTORE_PATH,
        'dataset_dir': DATASET_DIR,
        'env_name': ENV_NAME,
        'ckpt_num': CKPT_NUM,
        'replay_buffer_path': replay_buffer_path,
        'fql_train_steps': fql_train_steps,
        'fql_n_step': fql_n_step,
        'fql_discount': fql_discount,
        'fql_alpha': fql_alpha,
        'fql_best_of_n': fql_best_of_n,
        'seed': seed,
        'wandb_group': run_group + '_debug' if debug else run_group,
    }
    command = [
        'MUJOCO_GL=egl',
        f'python {run_file}',
        *(f'--{key}={value}' for key, value in flag_args.items()),
    ]
    return ' '.join(command)


def build_commands(debug):
    if debug:
        configs = [
            dict(
                replay_buffer_path=REPLAY_BUFFER_PATHS[0],
                fql_train_steps=50,
                fql_n_step=25,
                fql_discount=0.999,
                fql_alpha=100.0,
                fql_best_of_n=1,
                seed=1000,
            )
        ]
    else:
        seeds = [1000, 1001]
        best_of_n_values = [1, 4, 16]

        configs = []
        for seed, replay_buffer_path, fql_best_of_n in product(seeds, REPLAY_BUFFER_PATHS, best_of_n_values):
            configs.append(dict(
                replay_buffer_path=replay_buffer_path,
                fql_train_steps=1_000_000,
                fql_n_step=25,
                fql_discount=0.999,
                fql_alpha=100.0,
                fql_best_of_n=fql_best_of_n,
                seed=seed,
            ))

    return [make_command(debug=debug, **config) for config in configs]


for debug in [True, False]:
    commands = build_commands(debug)
    if not commands:
        raise ValueError(f'No runs generated for debug={debug}.')

    if args.gen == 'local':
        gen = LocalScriptGenerator(prefix=())
    else:
        gen = SbatchGenerator(
            j=args.num_jobs_per_gpu,
            limit=args.gpu_limit,
            prefix=(),
            comment=run_group,
            priority=priority,
        )
    gen.commands = commands

    generated = gen.generate_str()
    script_strs = [generated] if isinstance(generated, str) else generated
    name_prefix = f'{run_group}_{args.gen}_debug' if debug else f'{run_group}_{args.gen}'
    multi_part = len(script_strs) > 1

    for i, script_str in enumerate(script_strs, start=1):
        part_suffix = f'_part{i}' if multi_part else ''
        output_path = output_dir / f'{name_prefix}{part_suffix}.sh'
        with open(output_path, 'w') as f:
            f.write(script_str)
        print(f'Wrote {len(commands)} commands to {output_path}')
