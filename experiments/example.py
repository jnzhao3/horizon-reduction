
from generate import SbatchGenerator
from typing import NamedTuple

run_group = "qwq-2-13"
data_root = "/global/scratch/users/seohong/data/ogbench/"

num_jobs_per_gpu = 1
gpu_limit = 100


agent_params = dict(
    FQL={
        "puzzle-3x3-play-sparse":       dict(alpha=300.0),
        "scene-play-sparse":            dict(alpha=300.0),
        "cube-double-play":             dict(alpha=300.0),
        "antmaze-large-navigate":       dict(alpha=3.0), 
        "humanoidmaze-medium-navigate": dict(alpha=30.0),

        "cube-triple-play":             dict(alpha=30.0),
        "cube-quadruple-play":          dict(alpha=100.0),
        "puzzle-4x4-play-sparse":       dict(alpha=1.0),
        "antmaze-giant-navigate":       dict(alpha=3.0), 
        "humanoidmaze-large-navigate":  dict(alpha=30.0),
    },
)

for debug in [True, False]:
    gen = SbatchGenerator(j=num_jobs_per_gpu, limit=gpu_limit, prefix=("MUJOCO_GL=egl", "python main.py"), comment=run_group)
    if debug:
        gen.add_common_prefix({"run_group": run_group + "_debug", "offline_steps": 100, "eval_episodes": 1, "eval_interval": 5, "start_training": 50, "online_steps": 100, "log_interval": 25})
    else:
        gen.add_common_prefix({"run_group": run_group, "offline_steps": 1000000, "online_steps": 500000, "save_interval": 50000, "eval_interval": 50000})

    env_names = []
    domains = []
    for domain in [
        # "humanoidmaze-large-navigate",
        # "puzzle-4x4-play-sparse",
        # "humanoidmaze-medium-navigate",
        # "antmaze-giant-navigate", 
        "cube-triple-play",
        # "cube-quadruple-play",
        # "antmaze-large-navigate", 
        # "cube-double-play",
        # "scene-play-sparse",
        # "puzzle-3x3-play-sparse",
    ]:
        for task in [1, 2, 3, 4, 5]:
            if debug and task != 1: break
            if not debug and (("ant" in domain or "human" in domain) and task not in [1]): continue
            if not debug and (("cube" in domain or "puzzle" in domain or "scene" in domain) and task not in [2]): continue
            if debug and task != 1: break
            if domain.endswith("-sparse"):
                name = domain[:-7]
            else:
                name = domain
            env_names.append(f"{name}-singletask-task{task}-v0")
            domains.append(domain)

    for seed in [1001, 2002, 3003]:
        if debug and seed != 1001: break
        for env_name, domain in zip(env_names, domains):
            
            if "ant" in domain or "humanoid" in domain:
                horizon_length = 1
            else:
                horizon_length = 5 # action chunking of length 5 for all manipulation tasks
            
            base_kwargs = {
                "seed": seed,
                "utd_ratio": 1, 
                'agent.num_qs': 10,
                "env_name": env_name,
                "sparse": False if "sparse" not in domain else True,
                "horizon_length": horizon_length,
                "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
                "agent.action_chunking": True,
                "agent.batch_size": 256,
                "agent.rho": 0.0,
            }
            
            if "cube-quadruple" in env_name:
                base_kwargs["ogbench_dataset_dir"] = data_root + "cube-quadruple-play-100m-v0/"
            if "puzzle-4x4" in env_name:
                base_kwargs["ogbench_dataset_dir"] = data_root + "puzzle-4x4-play-100m-v0/"
            
            for gm_coef in [0., 0.0001, 0.001, 0.01, 0.1, 1.0]:
                # use FQL hparam otherwise
                kwargs = {"agent": "agents/old_qwq.py", "agent.gm_coef": gm_coef, **base_kwargs}
                for k, v in agent_params["FQL"][domain].items():
                    kwargs[f"agent.{k}"] = v
                orig_alpha = kwargs['agent.alpha']
                for alpha, name in zip([orig_alpha / 3, orig_alpha, orig_alpha * 3], ["d3", "orig", "m3"]):
                    kwargs["tags"] = f"QWQ,nr,gm={gm_coef},{name}"
                    kwargs["agent.alpha"] = alpha
                    gen.add_run(kwargs)

    sbatch_str_list = gen.generate_str()
    if debug:
        for index, sbatch_str in enumerate(sbatch_str_list):
            with open(f"sbatch/{run_group}-part{index+1}_debug.sh", "w") as f:
                f.write(sbatch_str)
    else:
        for index, sbatch_str in enumerate(sbatch_str_list):
            with open(f"sbatch/{run_group}-part{index+1}.sh", "w") as f:
                f.write(sbatch_str)
