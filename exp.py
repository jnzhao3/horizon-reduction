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
    "priority": "high", # high, normal, low, lowest
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
    "priority": "high", # high, normal, low, lowest
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
    "priority": "high", # high, normal, low, lowest
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

gcfql_maze_oracle3_1 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_1",
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
        "agent.train_goal_proposer" : True,
        "json_path": "../jsons/data.json",
    },
    "date": "2025-08-25"
}
gcfql_maze_oracle3_2 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_2",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 500000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.subgoal_steps" : 25,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_maze_oracle4 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 500000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.subgoal_steps" : 25,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_cube_oracle6 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 500000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.subgoal_steps" : 25,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_maze_oracle3_3 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_3",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.num_actions" : 4,
        "agent.num_qs" : 4,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_maze_oracle4_2 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_2",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.num_actions" : 4,
        "agent.num_qs" : 4,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_cube_oracle6_2 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_2",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.num_actions" : 4,
        "agent.num_qs" : 4,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-25"
}

gcfql_maze_oracle3_4 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_4",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-26"
}

gcfql_maze_oracle4_3 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_3",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-26"
}

gcfql_cube_oracle6_3 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_3",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-26"
}

gcfql_cube_oracle6_4 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_4",
        "env_name": "cube-double-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-double-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-27"
}

gcfql_cube_oracle6_5 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_5",
        "env_name": "cube-quadruple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-quadruple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0)
    },
    "date": "2025-08-27"
}

gcfql_maze_oracle3_5 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_5",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_maze_oracle4_4 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_4",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_cube_oracle6_6 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_6",
        "env_name": "cube-quadruple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-quadruple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 10,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.999
    },
    "date": "2025-08-27"
}

gcfql_maze_oracle3_6 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_6",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 50,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_maze_oracle4_5 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_5",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 50,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_cube_oracle6_7 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_7",
        "env_name": "cube-quadruple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-quadruple-play-v0",
        "train_data_size": 100000,
        "offline_steps": 2000000,
        "save_interval": 200000,
        "save_dir": "../../scratch/gcfql/",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 2,
        "agent.subgoal_steps" : 50,
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.999
    },
    "date": "2025-08-27"
}