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
run_group = "2026-03-31-00"
data_root = "../../scratch/data/"
output_dir = Path(__file__).resolve().parents[1] / "sbatch"
output_dir.mkdir(parents=True, exist_ok=True)
run_file = 'gc_gcfql_baseline.py'
priority = 'high'
##=========== END INITIAL INFORMATION ===========##

env_suffix = [f'-singletask-task{i}-v0' for i in range(1, 6)]
data_suffix = "-v0"

for debug in [True, False]:
    if args.gen == 'local':
        gen = LocalScriptGenerator(prefix=("MUJOCO_GL=egl", f"python {run_file}"))
    else:
        gen = SbatchGenerator(j=args.num_jobs_per_gpu, limit=args.gpu_limit, prefix=("MUJOCO_GL=egl", f"python {run_file}"), comment=run_group, priority=priority)
    if debug:
        gen.add_common_prefix({
            "run_group": run_group + "_debug",
            "offline_steps": 15,
            "further_offline_steps": 50,
            "collection_steps": 50,
            "eval_episodes": 0,
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
            "offline_steps": 1000000, # 1000000, # TODO: could possibly stop sooner?
            "further_offline_steps": 2000000,
            "collection_steps": 1000000,
            "save_interval": 100000,
            "eval_interval": 50000,
            "cleanup": True # TODO: might want to turn this off for others
        })

    env_names = []
    data_dirs = []
    for domain in [
        # "humanoidmaze-large-navigate",
        # "puzzle-4x4-play-sparse",
        # "humanoidmaze-medium-navigate",
        # "antmaze-medium-navigate",
        # "antmaze-giant-navigate", 
        "cube-triple-play",
        # "cube-quadruple-play",
        # "antmaze-large-navigate", 
        # "cube-double-play",
        # "scene-play-sparse",
        # "puzzle-3x3-play-sparse",
    ]:
        for suffix in env_suffix:
            env_names.append(f"{domain}{suffix}")
            data_dirs.append(f"{domain}{data_suffix}")
        # domains.append(domain)

    best_of_n = [4] # 4 seems like the best for humanoidmaze
    alphas = [300.0, 600.0, 1200.0]
    value_train = [True, False]
    use_policy_values = [True, False]

    run_count = 0
    for h in value_train:
        for n in best_of_n:
            for alpha in alphas:
                for use_policy in use_policy_values:
                    for seed in [1001, 2002, 3003]:
                        if debug and seed != 1001: break
                        for env_name, data_dir in zip(env_names, data_dirs):

                            base_kwargs = {
                                "seed": seed,
                                "env_name": env_name,
                                "dataset_dir": data_root + data_dir,
                                "agent": "agents/gcfql.py",
                                "agent.train_goal_proposer": False,
                                "agent.goal_proposer_type": "default",
                                "agent.train_value": h,
                                "agent.horizon_length": 25,
                                "agent.action_chunking": True if 'cube' in env_name or 'puzzle' in env_name else False,
                                "agent.num_qs": 10,
                                "agent.q_agg": "mean",
                                "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                                "agent.batch_size": 256,
                                "agent.alpha": alpha,
                                # "agent.subgoal_steps": h,
                                "agent.awr_invtemp": 0.0,
                                "agent.best_of_n": n,
                                "agent.use_policy_for_value": use_policy,
                                "fql_agent": "agents/fql.py",
                                "fql_agent.best_of_n": n,
                                "fql_agent.horizon_length": 25,
                                "fql_agent.action_chunking": True if 'cube' in env_name or 'puzzle' in env_name else False,
                                "fql_agent.num_qs": 10,
                                "fql_agent.q_agg": "mean",
                                "fql_agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                                "fql_agent.batch_size": 256,
                                "fql_agent.alpha": alpha,
                            }

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
