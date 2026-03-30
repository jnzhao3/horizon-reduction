from generate_local import LocalScriptGenerator
from generate import SbatchGenerator
from pathlib import Path
from argparse import ArgumentParser

##=========== BEGIN ARGUMENTS ===========##
parser = ArgumentParser()
parser.add_argument('--gen', type=str, default='brc', help='where to run the script')
parser.add_argument('--num_jobs_per_gpu', type=int, default=1, help='the number of jobs to allocate per gpu')
parser.add_argument('--gpu_limit', type=int, default=100)
args = parser.parse_args()
##=========== END ARGUMENTS ===========##

##=========== INITIAL INFORMATION ===========##
run_group = "2026-03-29-00"
data_root = "../../scratch/data/"
output_dir = Path(__file__).resolve().parents[1] / "sbatch"
output_dir.mkdir(parents=True, exist_ok=True)
run_file = 'train_value.py'
priority = 'high'
##=========== END INITIAL INFORMATION ===========##

agent_params = dict(
    GCFQL={
        "puzzle-3x3-play-sparse":       dict(alpha=3.0),
        "scene-play-sparse":            dict(alpha=3.0),
        "cube-double-play":             dict(alpha=3.0),
        "antmaze-large-navigate":       dict(alpha=3.0),
        "humanoidmaze-medium-navigate": dict(alpha=3.0),
        "antmaze-medium-navigate":      dict(alpha=3.0),
        "cube-triple-play":             dict(alpha=3.0),
        "cube-quadruple-play":          dict(alpha=3.0),
        "puzzle-4x4-play-sparse":       dict(alpha=3.0),
        "antmaze-giant-navigate":       dict(alpha=3.0),
        "humanoidmaze-large-navigate":  dict(alpha=3.0),
    },
)

env_suffix = [f'-singletask-task{i}-v0' for i in range(3, 5)] # changing to only tasks 3 and 4
data_suffix = "-v0"

for debug in [True, False]:
    if args.gen == 'local':
        gen = LocalScriptGenerator(prefix=("MUJOCO_GL=egl", f"python {run_file}"))
    else:
        gen = SbatchGenerator(j=args.num_jobs_per_gpu, limit=args.gpu_limit, prefix=("MUJOCO_GL=egl", f"python {run_file}"), comment=run_group, priority=priority)
    if debug:
        gen.add_common_prefix({
            "run_group": run_group + "_debug",
            "offline_steps": 100,
            "eval_episodes": 1,
            "video_episodes": 0,
            "eval_interval": 50,
            "save_interval": 100,
            "log_interval": 10,
            "cleanup": True
        })
    else:
        gen.add_common_prefix({
            "run_group": run_group,
            "offline_steps": 100000,
            "eval_interval": 10000,
            "save_interval": 50000,
            "log_interval": 1000,
            "cleanup": True
        })

    env_names = []
    data_dirs = []
    domains = []
    for domain in [
        "humanoidmaze-medium-navigate",
        # "cube-triple-play",
        # "antmaze-medium-navigate",
    ]:
        for suffix in env_suffix:
            env_names.append(f"{domain}{suffix}")
            data_dirs.append(f"{domain}{data_suffix}")
        domains.append(domain)

    run_count = 0
    for seed in [1001, 2002, 3003]:
        if debug and seed != 1001: break
        for env_name, data_dir in zip(env_names, data_dirs):

            base_kwargs = {
                "seed": seed,
                "env_name": env_name,
                "dataset_dir": data_root + data_dir,

                "agent": "agents/gcfql.py",
                "agent.train_value": True,
                "agent.horizon_length": 25,
                "agent.action_chunking": True if 'cube' in env_name or 'puzzle' in env_name else False,
                "agent.num_qs": 10,
                "agent.q_agg": "min",
                "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                "agent.batch_size": 256,
                "agent.alpha": 3.0,
                "agent.critic_loss_type": "bce",
            }

            if "cube-quadruple" in env_name:
                base_kwargs["dataset_dir"] = data_root + "cube-quadruple-play-100m-v0"
            if "puzzle-4x4" in env_name:
                base_kwargs["dataset_dir"] = data_root + "puzzle-4x4-play-100m-v0"

            gen.add_run(base_kwargs)
            run_count += 1

    if run_count == 0:
        raise ValueError(f"No runs generated for debug={debug}.")

    ##=========== GENERATE ===========##
    generated = gen.generate_str()
    script_strs = [generated] if isinstance(generated, str) else generated
    name_prefix = f"{run_group}_{args.gen}_debug" if debug else f"{run_group}_{args.gen}"
    multi_part = len(script_strs) > 1

    for i, script_str in enumerate(script_strs, start=1):
        part_suffix = f"_part{i}" if multi_part else ""
        with open(output_dir / f"{name_prefix}{part_suffix}.sh", "w") as f:
            f.write(script_str)
            print(script_str)