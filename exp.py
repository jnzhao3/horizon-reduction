# python3 main.py --env_name=cube-triple-play-oraclerep-v0 --agent=../agents/gcfql.py --dataset_dir ../../../scratch/data/cube-triple-play-v0 --log_interval 100000 --save_interval 1000000 --save_dir ../../../scratch/gcfql/ --agent.alpha 300 --agent.actor_type default'

# export PYTHONPATH="../:${PYTHONPATH}"


debug = {
    "script": "main.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "12:00:00",
    "config": {
        "run_group": "debugging",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": 100,
        "offline_steps": 100,
        "log_interval": 50,
        "save_interval": 50,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "json_path": "../jsons/data.json",
    },
    "date": "2025-08-16"
}

gcfql_cube_oracle5 = {
    "script": "main.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle5",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": 100000,
        # "offline_steps": 100,
        # "log_interval": 50,
        # "save_interval": 50,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "json_path": "../jsons/data.json",
    },
    "date": "2025-08-16"
}

gcfql_maze_oracle1 = {
    "script": "main.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle1",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoid-medium-navigate-v0",
        "train_data_size": 100000,
        # "offline_steps": 100,
        # "log_interval": 50,
        # "save_interval": 50,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "json_path": "../jsons/data.json",
    },
    "date": "2025-08-16"
}

gcfql_maze_oracle2 = {
    "script": "main.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle2",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        # "offline_steps": 100,
        # "log_interval": 50,
        # "save_interval": 50,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "json_path": "../jsons/data.json",
    },
    "date": "2025-08-16"
}