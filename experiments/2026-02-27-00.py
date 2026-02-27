from generate_local import LocalScriptGenerator
from generate import SbatchGenerator
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--gen', type=str, default='brc', help='where to run the script')
parser.add_argument('--num_jobs_per_gpu', type=int, default=1, help='the number of jobs to allocate per gpu')
parser.add_argument('--gpu_limit', type=int, default=100)
args = parser.parse_args()

run_group = "2026-02-27-00"
# data_root = "/global/scratch/users/seohong/data/ogbench/"
data_root = "../../scratch/data/"
output_dir = Path(__file__).resolve().parents[1] / "sbatch"
output_dir.mkdir(parents=True, exist_ok=True)

agent_params = dict(
    FQL={
        "puzzle-3x3-play-sparse":       dict(alpha=300.0),
        "scene-play-sparse":            dict(alpha=300.0),
        "cube-double-play":             dict(alpha=300.0),
        "antmaze-large-navigate":       dict(alpha=3.0), 
        "humanoidmaze-medium-navigate": dict(alpha=30.0),
        "antmaze-medium-navigate":      dict(alpha=3.0), # TODO: check if this one is correct
        "cube-triple-play":             dict(alpha=30.0),
        "cube-quadruple-play":          dict(alpha=100.0),
        "puzzle-4x4-play-sparse":       dict(alpha=1.0),
        "antmaze-giant-navigate":       dict(alpha=3.0), 
        "humanoidmaze-large-navigate":  dict(alpha=30.0),
    },
)

env_suffix = "-oraclerep-v0"
data_suffix = "-v0"
train_data_size = 100000

for debug in [True, False]:
    if args.gen == 'local':
        gen = LocalScriptGenerator(prefix=("MUJOCO_GL=egl", "python e2e.py"))
    else:
        # gen = SbatchGenerator(prefix=("MUJOCO_GL=egl", "python e2e.py"))
        gen = SbatchGenerator(j=args.num_jobs_per_gpu, limit=args.gpu_limit, prefix=("MUJOCO_GL=egl", "python e2e.py"), comment=run_group)
    if debug:
        gen.add_common_prefix({
            "run_group": run_group + "_debug",
            "offline_steps": 50,
            "collection_steps": 50,
            "eval_episodes": 1,
            "video_episodes": 0,
            "eval_interval": 5,
            "save_interval": 25,
            "data_plot_interval": 25,
            "log_interval": 25,
            "cleanup": True
        })
    else:
        gen.add_common_prefix({
            "run_group": run_group,
            "offline_steps": 200000, # 1000000, # TODO: could possibly stop sooner?
            "collection_steps": 1000000,
            "save_interval": 100000,
            "eval_interval": 50000,
        })

    env_names = []
    data_dirs = []
    domains = []
    for domain in [
        # "humanoidmaze-large-navigate",
        # "puzzle-4x4-play-sparse",
        "humanoidmaze-medium-navigate",
        "antmaze-medium-navigate",
        # "antmaze-giant-navigate", 
        # "cube-triple-play",
        # "cube-quadruple-play",
        # "antmaze-large-navigate", 
        # "cube-double-play",
        # "scene-play-sparse",
        # "puzzle-3x3-play-sparse",
    ]:
        # for task in [1, 2, 3, 4, 5]:
        #     if debug and task != 1: break
        #     if not debug and (("ant" in domain or "human" in domain) and task not in [1]): continue
        #     if not debug and (("cube" in domain or "puzzle" in domain or "scene" in domain) and task not in [2]): continue
        #     if debug and task != 1: break
        #     if domain.endswith("-sparse"):
        #         name = domain[:-7]
        #     else:
        #         name = domain
        env_names.append(f"{domain}{env_suffix}")
        data_dirs.append(f"{domain}{data_suffix}")
        domains.append(domain)

    run_count = 0
    for seed in [1001, 2002, 3003]:
        if debug and seed != 1001: break
        for env_name, data_dir, domain in zip(env_names, data_dirs, domains):
            
            base_kwargs = {
                "seed": seed,
                "env_name": env_name,
                "dataset_dir": data_root + data_dir,
                "agent": "../agents/gcfql.py",
                "wrapper": "wrappers/rndsubgoals.py",
                "wrapper.max_episode_steps": 2000,
                "wrapper.pre_init": True,
                "agent.actor_type": "best-of-n",
                "agent.train_goal_proposer": True,
                "agent.goal_proposer_type": "actor-gc",
                "agent.num_qs": 10,
                "agent.q_agg": "mean",
                "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                "agent.batch_size": 256,
                "agent.alpha": agent_params["FQL"][domain]["alpha"],
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
