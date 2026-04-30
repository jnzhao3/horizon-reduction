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


run_group = '2026-04-29-05'
output_dir = Path(__file__).resolve().parents[1] / 'sbatch'
output_dir.mkdir(parents=True, exist_ok=True)
run_file = '19_data_collection_cube_safe.py'
priority = 'normal'

RESTORE_PATH = '../../scratch/2026-04-27-dqc/dqc/2026-04-27-dqc/sd200002s_33839966.0.33839966.2.20260427_210028/'
DATASET_PATH = '../../scratch/data/cube-quadruple-play-v0/cube-quadruple-play-v0.npz'
DATASET_DIR = '../../scratch/data/cube-quadruple-play-v0/'
FLOW_RESTORE_PATH = '../../scratch/checkpoints/cube_quadruple_horizon_subgoal_proposer'

ENV_NAME = 'cube-quadruple-play-oraclerep-v0'
CKPT_NUM = 1_000_000
FLOW_CKPT_NUM = 1_050_000


def make_command(
    *,
    task_id,
    subgoal_steps,
    steps_to_subgoal,
    num_additional_steps,
    fql_train_steps,
    fql_chunk_size,
    fql_n_step,
    fql_discount,
    fql_alpha,
    num_subgoals,
    mult_factor,
    additive_factor,
    a_b_factor,
    b_c_factor,
    seed,
    debug,
    dataset_path=None,
    dataset_dir=None,
    dataset_replace_interval=1000,
):
    flag_args = {
        'restore_path': RESTORE_PATH,
        'flow_restore_path': FLOW_RESTORE_PATH,
        'flow_ckpt_num': FLOW_CKPT_NUM,
        'env_name': ENV_NAME,
        'ckpt_num': CKPT_NUM,
        'task_id': task_id,
        'subgoal_steps': subgoal_steps,
        'steps_to_subgoal': steps_to_subgoal,
        'num_additional_steps': num_additional_steps,
        'fql_train_steps': fql_train_steps,
        'fql_chunk_size': fql_chunk_size,
        'fql_n_step': fql_n_step,
        'fql_discount': fql_discount,
        'fql_alpha': fql_alpha,
        'num_subgoals': num_subgoals,
        'mult_factor': mult_factor,
        'additive_factor': additive_factor,
        'A_B_factor': a_b_factor,
        'B_C_factor': b_c_factor,
        'dataset_replace_interval': dataset_replace_interval,
        'seed': seed,
        'wandb_group': run_group + '_debug' if debug else run_group,
    }
    if dataset_dir is not None:
        flag_args['dataset_dir'] = dataset_dir
    else:
        flag_args['dataset_path'] = dataset_path or DATASET_PATH
    command = [
        'MUJOCO_GL=egl',
        f'python {run_file}',
        *(f'--{key}={value}' for key, value in flag_args.items()),
    ]
    return ' '.join(command)


def build_commands(debug):
    if debug:
        configs = [
            # Basic debug run with single dataset.
            dict(
                task_id=1,
                subgoal_steps=250,
                steps_to_subgoal=25,
                num_additional_steps=50,
                fql_train_steps=50,
                fql_chunk_size=5,
                fql_n_step=25,
                fql_discount=0.999,
                fql_alpha=300.0,
                num_subgoals=128,
                mult_factor=1.0,
                additive_factor=0.0,
                a_b_factor=1.0,
                b_c_factor=0.0,
                seed=1000,
                dataset_path=DATASET_PATH,
                dataset_replace_interval=1000,
            ),
            # Dataset cycling debug run: replace every 10 steps so cycling triggers within 50.
            dict(
                task_id=1,
                subgoal_steps=250,
                steps_to_subgoal=25,
                num_additional_steps=50,
                fql_train_steps=50,
                fql_chunk_size=5,
                fql_n_step=25,
                fql_discount=0.999,
                fql_alpha=300.0,
                num_subgoals=128,
                mult_factor=1.0,
                additive_factor=0.0,
                a_b_factor=1.0,
                b_c_factor=0.0,
                seed=1000,
                dataset_dir=DATASET_DIR,
                dataset_replace_interval=10,
            ),
        ]
    else:
        seeds = [1000, 1001]
        task_ids = [1, 2, 3, 4, 5]
        horizons = [100, 250]
        mult_factors = [0.9, 1.0]

        configs = []
        for seed, task_id, horizon, mult_factor in product(seeds, task_ids, horizons, mult_factors):
            configs.append(dict(
                task_id=task_id,
                subgoal_steps=horizon,
                steps_to_subgoal=25,
                num_additional_steps=1_000_000,
                fql_train_steps=1_000_000,
                fql_chunk_size=5,
                fql_n_step=25,
                fql_discount=0.999,
                fql_alpha=300.0,
                num_subgoals=128,
                mult_factor=mult_factor,
                additive_factor=0.0,
                a_b_factor=1.0,
                b_c_factor=0.0,
                seed=seed,
                dataset_path=DATASET_PATH,
                dataset_replace_interval=1000,
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
