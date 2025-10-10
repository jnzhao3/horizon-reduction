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

gcfql_maze_oracle3_7 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_7",
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
        "agent.subgoal_steps" : (50,75,100),
        "agent.value_loss_type" : "squared", # not necessary
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_maze_oracle4_6 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_6",
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
        "agent.subgoal_steps" : (50,75,100),
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-26"
}

gcfql_cube_oracle6_8 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_8",
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
        "agent.subgoal_steps" : (50,75,100),
        "agent.value_loss_type" : "squared",
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.999
    },
    "date": "2025-08-27"
}

gcfql_maze_oracle4_7 = { # Debugging script
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_7",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 1000000,
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
        "agent.subgoal_steps" : (50,75,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": (0.0,1.0),
        "agent.discount" : 0.995
    },
    "date": "2025-08-29"
}

gcfql_maze_oracle3_8 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle3_8",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : ("awr","default","actor-gc")
    },
    "date": "2025-08-26"
}

gcfql_maze_oracle4_8 = {
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_maze_oracle4_8",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : ("awr","default","actor-gc")
    },
    "date": "2025-08-29"
}

gcfql_cube_oracle6_9 = { 
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle6_9",
        "env_name": "cube-quadruple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-quadruple-play-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.999,
        "agent.goal_proposer_type" : ("awr","default","actor-gc")
    },
    "date": "2025-08-29"
}
gcfql_cube_oracle7_1 = { 
    "script": "main.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "gcfql_cube_oracle7_1",
        "env_name": "cube-triple-play-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/cube-triple-play-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.999,
        "agent.goal_proposer_type" : ("awr","default","actor-gc")
    },
    "date": "2025-09-01"
}

train_value1_1 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value1_1",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            ("../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1402547812974.s_27684700_4.20250830_020259_647.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1414568005230.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1432255779435.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1433302649454.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1434207663726.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1436102206062.s_27684113_3.20250829_220125_148.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1438068558443.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1444406050411.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1447825734251.s_27684113_3.20250829_220125_147.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448582269550.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448722333294.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8"),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample"),
        "offline_steps": 100,
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,
    },
    "date": "2025-09-07"
}

train_value1_2 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value1_2",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            ("../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1402547812974.s_27684700_4.20250830_020259_647.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1414568005230.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1432255779435.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1433302649454.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1434207663726.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1436102206062.s_27684113_3.20250829_220125_148.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1438068558443.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1444406050411.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1447825734251.s_27684113_3.20250829_220125_147.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448582269550.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448722333294.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8"),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample")
    },
    "date": "2025-09-07"
}

train_value1_3 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value1_3",
        "env_name": "humanoidmaze-large-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-large-navigate-v0",
        # "train_data_size": (100000,1000000),
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
        # "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            ("../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1402547812974.s_27684700_4.20250830_020259_647.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1414568005230.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1432255779435.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1433302649454.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1434207663726.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1436102206062.s_27684113_3.20250829_220125_148.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1438068558443.s_27684103_6.20250830_020534_012.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1444406050411.s_27684111_1.20250829_220124_123.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1447825734251.s_27684113_3.20250829_220125_147.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448582269550.s_27684707_5.20250830_020510_865.gcfql_maze_oracle3_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle3_8/sd000_1448722333294.s_27684112_2.20250829_220125_578.gcfql_maze_oracle3_8"),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample")
    },
    "date": "2025-09-07"
}

train_value2_1 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value2_1",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            (
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1456162036334.s_27680590_6.20250829_185905_857.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1453467320942.s_27680626_4.20250829_185906_336.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448342040171.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1438812371566.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1436627854958.s_27680618_2.20250829_185820_547.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1426623563374.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414600722027.s_27680590_6.20250829_185845_686.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414321273454.s_27680627_5.20250829_185845_775.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1452282852974.s_27680618_2.20250829_185841_310.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448103063147.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1441449117294.s_27680626_4.20250829_185845_562.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1433843123822.s_27680627_5.20250829_185906_465.gcfql_maze_oracle4_8"
        ),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample"),
        "offline_steps": 100,
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,
    },
    "date": "2025-09-07"
}

train_value2_2 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value2_2",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000,1000000),
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
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            (
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1456162036334.s_27680590_6.20250829_185905_857.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1453467320942.s_27680626_4.20250829_185906_336.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448342040171.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1438812371566.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1436627854958.s_27680618_2.20250829_185820_547.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1426623563374.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414600722027.s_27680590_6.20250829_185845_686.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414321273454.s_27680627_5.20250829_185845_775.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1452282852974.s_27680618_2.20250829_185841_310.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448103063147.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1441449117294.s_27680626_4.20250829_185845_562.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1433843123822.s_27680627_5.20250829_185906_465.gcfql_maze_oracle4_8"
        ),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample"),
    #     "offline_steps": 100,
    #     "log_interval": 10,
    #     "eval_interval": 50,
    #     "save_interval": 100,
    },
    "date": "2025-09-07"
}

train_value2_3 = { 
    "script": "train_value.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "train_value2_3",
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        # "train_data_size": (100000,1000000),
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
        # "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.awr_invtemp": 1.0,
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("awr","default","actor-gc")
        # "restore_path": "../../scratch/gcfql/gcfql_maze_oracle3_8_2025-08-26_18-37-28/model_final.pt",
        "restore_path" : 
            (
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1456162036334.s_27680590_6.20250829_185905_857.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1453467320942.s_27680626_4.20250829_185906_336.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448342040171.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1438812371566.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1436627854958.s_27680618_2.20250829_185820_547.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1426623563374.s_27680619_3.20250829_185930_100.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414600722027.s_27680590_6.20250829_185845_686.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1414321273454.s_27680627_5.20250829_185845_775.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1452282852974.s_27680618_2.20250829_185841_310.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1448103063147.s_27680617_1.20250829_190015_646.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1441449117294.s_27680626_4.20250829_185845_562.gcfql_maze_oracle4_8",
            "../../scratch/gcfql/horizon-reduction/gcfql_maze_oracle4_8/sd000_1433843123822.s_27680627_5.20250829_185906_465.gcfql_maze_oracle4_8"
        ),
        "restore_epoch": 2000000,
        "q_pred_calc" : ("batch", "sample"),
    #     "offline_steps": 100,
    #     "log_interval": 10,
    #     "eval_interval": 50,
    #     "save_interval": 100,
    },
    "date": "2025-09-07"
}

e2e_maze_1_1 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "e2e_maze_1_1",
        "seed": (0, 1),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 10,
        "eval_interval": 50,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100),
        "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : 
                "'aorl/gcfql_main_maze3_5/sd000_1460317626035.s_27767445_102.20250903_082118_234.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd001_1462948537011.s_27768212_222.20250903_103200_215.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd000_1436076103347.s_27767225_78.20250903_072451_934.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd001_1433018375859.s_27768014_198.20250903_100607_083.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd000_1410219265718.s_27767146_54.20250903_062514_370.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd001_1423283553971.s_27767875_174.20250903_095433_086.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd000_1453816609459.s_27767113_30.20250903_055039_644.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd001_1461153812147.s_27767741_150.20250903_093412_478.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd000_1416582666931.s_27766912_6.20250903_045337_719.gcfql_main_maze3_5/data-2000000.npz','aorl/gcfql_main_maze3_5/sd001_1437168725686.s_27767521_126.20250903_085950_906.gcfql_main_maze3_5/data-2000000.npz'",
        "data_option.train_data_sizes" : "100000,100000,100000,100000,100000,100000,100000,100000,100000,100000",
    }
}
e2e_maze_1_2 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "e2e_maze_1_2",
        "seed": (0, 1),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 1,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : "\"\"aorl/gcfql_main_maze3_5/sd000_1460317626035.s_27767445_102.20250903_082118_234.gcfql_main_maze3_5/data-2000000.npz\",\"aorl/gcfql_main_maze3_5/sd001_1462948537011.s_27768212_222.20250903_103200_215.gcfql_main_maze3_5/data-2000000.npz\"\"",
        "data_option.train_data_sizes" : "100000,100000",
    }
}

e2e_maze_1_3 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "e2e_maze_1_3",
        "seed": (0, 1),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
         "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : "debug",
        "data_option.train_data_sizes" : ("100000,100000"),
    }
}

e2e_maze_1_4 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "36:00:00",
    "config": {
        "run_group": "e2e_maze_1_4",
        "seed": (0, 1),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 2000000,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : "gcfql_main_maze3_5",
        "data_option.train_data_sizes" : ("100000,100000,100000,100000,100000,100000,100000,100000,100000,100000","10000,10000,10000,10000,10000,10000,10000,10000,10000,10000"),
    }
}

e2e_maze_1_5 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "16:00:00",
    "config": {
        "run_group": "e2e_maze_1_5",
        "seed": (0),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 2000000,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : "gcfql_main_maze3_5",
        "data_option.train_data_sizes" : ("100000,100000,100000,100000,100000,100000,100000,100000,100000,100000"),
    }
}

e2e_maze_1_6 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "16:00:00",
    "config": {
        "run_group": "e2e_maze_1_6",
        "seed": (0),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 2000000,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/combine_with.py",
        "data_option.train_data_paths" : "gcfql_main_maze3_5",
        "data_option.train_data_sizes" : ("100000,100000,100000,100000,100000,100000,100000,100000,100000,100000"),
    }
}

e2e_maze_1_7 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "16:00:00",
    "config": {
        "run_group": "e2e_maze_1_7",
        "seed": (0),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/randomsteps.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
    }
}

e2e_maze_1_8 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "1:00:00",
    "config": {
        "run_group": "e2e_maze_1_8",
        "seed": (0),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": 100000,
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        # "agent.goal_proposer_type" : ("default","actor-gc"),
        "data_option" : "datafuncs/ogbench.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
        "data_option.noise": (0.0, 0.1) # TODO: reference what the actual values are
    }
}

e2e_maze_1_9 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "16:00:00",
    "config": {
        "run_group": "e2e_maze_1_9",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/ogbench.py",
        "data_option.save_data_interval" : 50,
        "data_option.noise": 0.0 # TODO: reference what the actual values are
    }
}

e2e_maze_1_10 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_1_10",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/ogbench.py",
    }
}

e2e_maze_2_1 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_2_1",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.goal_proposer_type" : "default",
        "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withsubgoal.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
    }
}

e2e_maze_2_2 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "10:00:00",
    "config": {
        "run_group": "e2e_maze_2_2",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000,
        "eval_episodes": 5,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.goal_proposer_type" : "default",
        "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withsubgoal.py",
    }
}

e2e_maze_2_3 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_2_3",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000000,
        "eval_episodes": 5,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.goal_proposer_type" : "default",
        "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withsubgoal.py",
    }
}

e2e_maze_3_1 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_1",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
        "debug": "=True"
    }
}

e2e_maze_3_2 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "10:00:00",
    "config": {
        "run_group": "e2e_maze_3_2",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
    }
}

e2e_maze_3_3 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_3",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000000,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
    }
}

e2e_maze_4_1 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_4_1",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
        "debug": "=True"
    }
}

e2e_maze_4_2 = {
    "script": "e2e.py",
    "priority": "low", # high, normal, low, lowest
    "time": "1:00:00",
    "config": {
        "run_group": "e2e_maze_4_2",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
    }
}

e2e_maze_4_3 = {
    "script": "e2e.py",
    "priority": "low", # high, normal, low, lowest
    "time": "1:00:00",
    "config": {
        "run_group": "e2e_maze_4_3",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.goal_proposer_type" : "default",
        "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withsubgoal.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
    }
}

e2e_maze_4_4 = {
    "script": "e2e.py",
    "priority": "low", # high, normal, low, lowest
    "time": "1:00:00",
    "config": {
        "run_group": "e2e_maze_4_4",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 100,
        "save_interval": 50,
        "log_interval": 100,
        "eval_interval": 50,
        "eval_episodes": 0,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/ogbench.py",
        "data_option.collection_steps" : 100,
        "data_option.save_data_interval" : 50,
        "data_option.plot_interval" : 50,
        "data_option.max_episode_steps": 20,
    }
}

# actual runs

e2e_maze_4_5 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_4_5",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withrnd.py",
    }
}

e2e_maze_4_6 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_4_6",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.goal_proposer_type" : "default",
        "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/withsubgoal.py",
    }
}

e2e_maze_4_7 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_4_7",
        "seed": (0,1,2,3),
        "env_name": "antmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/antmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        # "agent.goal_proposer_type" : "default",
        # "agent.subgoal_steps" : (50,100,200),
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/ogbench.py",
    }
}

e2e_maze_3_4 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_3",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000,
        "eval_episodes": 0,

        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/rndandsubgoal.py",
    }
}

e2e_maze_3_4 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_4",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "offline_steps": 1000,
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/rndandsubgoal.py",
    }
}

e2e_maze_3_5 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_5",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "video_episodes": 0,
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/rndandsubgoal.py",
    }
}

e2e_maze_3_6 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_3_6",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
        # "json_path": "../jsons/data.json",
        "agent.discount" : 0.995,
        "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.discount" : 0.995,
        "data_option" : "datafuncs/rndandsubgoal.py",
    }
}

e2e_maze_5_1 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_5_1",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/withrnd.py",
        "wrapper.max_episode_steps": 5,
        "offline_steps": 10,
        "collection_steps": 10,
        "data_plot_interval": 5,
        "log_interval": 5,
        "eval_interval": 5,
        "save_interval": 5,
        "eval_episodes": 0,
        "video_episodes": 0,
    }
}

e2e_maze_5_2 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_5_2",
        "seed": (0,1,2,3,4,5),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/withrnd.py",
        "wrapper.max_episode_steps": 2000,
        "offline_steps": (100000, 1000000),
        "collection_steps": 1000000,
        "data_plot_interval": 5,
        "log_interval": 5,
        "eval_interval": 5,
        "save_interval": 5,
        "eval_episodes": 0,
        "video_episodes": 1,
    }
}

e2e_maze_5_3 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_5_3",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/withrnd.py",
        "wrapper.max_episode_steps": 2000,
        "offline_steps": (100000, 1000000),
        "collection_steps": 1000000,
        "video_episodes": 1,
    }
}

e2e_maze_6_1 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_6_1",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/randomsteps.py",
        # "wrapper.max_episode_steps": 2000,
        # "offline_steps": (100000, 1000000),
        # "collection_steps": 1000000,
        # "video_episodes": 1,
        "wrapper.max_episode_steps": 5,
        "offline_steps": 10,
        "collection_steps": 10,
        "data_plot_interval": 5,
        "log_interval": 5,
        "eval_interval": 5,
        "save_interval": 5,
        "eval_episodes": 0,
        "video_episodes": 0,
    }
}

e2e_maze_6_2 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_6_2",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=False",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        # "agent.subgoal_steps" : (50,100,200),
    #    "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/randomsteps.py",
        "wrapper.max_episode_steps": 2000,
        "offline_steps": (100000, 1000000),
        "collection_steps": 1000000,
        "video_episodes": 1,
        # "wrapper.max_episode_steps": 5,
        # "offline_steps": 10,
        # "collection_steps": 10,
        # "data_plot_interval": 5,
        # "log_interval": 5,
        # "eval_interval": 5,
        # "save_interval": 5,
        # "eval_episodes": 0,
        # "video_episodes": 0,
    }
}

e2e_maze_7_1 = {
    "script": "e2e.py",
    "priority": "normal", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_7_1",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/rndsubgoals.py",
        # "wrapper.max_episode_steps": 2000,
        # "offline_steps": (100000, 1000000),
        # "collection_steps": 1000000,
        # "video_episodes": 1,
        "wrapper.max_episode_steps": 5,
        "offline_steps": 10,
        "collection_steps": 10,
        "data_plot_interval": 5,
        "log_interval": 5,
        "eval_interval": 5,
        "save_interval": 5,
        "eval_episodes": 0,
        "video_episodes": 0,
    }
}

e2e_maze_7_2 = {
    "script": "e2e.py",
    "priority": "high", # high, normal, low, lowest
    "time": "24:00:00",
    "config": {
        "run_group": "e2e_maze_7_2",
        "seed": (0,1,2,3),
        "env_name": "humanoidmaze-medium-navigate-oraclerep-v0", # use oracle representation!
        "agent": "../agents/gcfql.py",
        "dataset_dir": "../../scratch/data/humanoidmaze-medium-navigate-v0",
        "train_data_size": (100000, 1000000),
        "save_dir": "../../scratch",
        "agent.alpha": 300,
        "agent.actor_type": "best-of-n",
        "agent.train_goal_proposer" : "=True",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.batch_size" : 256,
        "agent.num_actions" : 8,
        "agent.num_qs" : 10,
        "agent.q_agg" : "mean",
        "agent.subgoal_steps" : (50,100,200),
       "agent.goal_proposer_type" : "actor-gc",
        "agent.actor_hidden_dims" : "512,512,512,512",
        "agent.value_hidden_dims" : "512,512,512,512",
        "agent.discount" : 0.995,
        "wrapper" : "wrappers/rndsubgoals.py",
        "wrapper.max_episode_steps": 2000,
        "offline_steps": 1000000,
        "collection_steps": 1000000,
        "video_episodes": 1,
        # "wrapper.max_episode_steps": 5,
        # "offline_steps": 10,
        # "collection_steps": 10,
        # "data_plot_interval": 5,
        # "log_interval": 5,
        # "eval_interval": 5,
        # "save_interval": 5,
        # "eval_episodes": 0,
        # "video_episodes": 0,
    }
}