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


run_group = '2026-04-28-00'
output_dir = Path(__file__).resolve().parents[1] / 'sbatch'
output_dir.mkdir(parents=True, exist_ok=True)
run_file = '10_data_collection_experiment.py'
priority = 'high'

RESTORE_PATH = '../../scratch/dqc-reproduce/sd100001s_33415523.0.33415522.1.20260415_020458/'
DATASET_PATH = '../../scratch/data/humanoidmaze-giant-navigate-v0/humanoidmaze-giant-navigate-100m-v0/humanoidmaze-giant-navigate-v0-003.npz'
ENV_NAME = 'humanoidmaze-giant-navigate-v0'
CKPT_NUM = 1_000_000


def make_command(
    *,
    task_id,
    subgoal_steps,
    steps_to_subgoal,
    num_train_steps,
    num_additional_steps,
    fql_train_steps,
    num_subgoals,
    mult_factor,
    additive_factor,
    a_b_factor,
    b_c_factor,
    seed,
    debug,
):
    flag_args = {
        'restore_path': RESTORE_PATH,
        'dataset_path': DATASET_PATH,
        'env_name': ENV_NAME,
        'ckpt_num': CKPT_NUM,
        'task_id': task_id,
        'subgoal_steps': subgoal_steps,
        'steps_to_subgoal': steps_to_subgoal,
        'num_train_steps': num_train_steps,
        'num_additional_steps': num_additional_steps,
        'fql_train_steps': fql_train_steps,
        'num_subgoals': num_subgoals,
        'mult_factor': mult_factor,
        'additive_factor': additive_factor,
        'A_B_factor': a_b_factor,
        'B_C_factor': b_c_factor,
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
                task_id=1,
                subgoal_steps=100,
                steps_to_subgoal=25,
                num_train_steps=50,
                num_additional_steps=50,
                fql_train_steps=50,
                num_subgoals=16,
                mult_factor=0.9,
                additive_factor=0.0,
                a_b_factor=1.0,
                b_c_factor=0.0,
                seed=1000,
            )
        ]
    else:
        seeds = [1000, 1001]
        subgoal_steps_list = [100, 1000]
        mult_factors = [0.9, 1.0]

        configs = []
        for seed, subgoal_steps, mult_factor in product(seeds, subgoal_steps_list, mult_factors):
            configs.append(dict(
                task_id=1,
                subgoal_steps=subgoal_steps,
                steps_to_subgoal=25,
                num_train_steps=3_000_000,
                num_additional_steps=1_000_000,
                fql_train_steps=1_000_000,
                num_subgoals=128,
                mult_factor=mult_factor,
                additive_factor=0.0,
                a_b_factor=1.0,
                b_c_factor=0.0,
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
