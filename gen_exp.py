import os
import argparse
from platform import python_branch
import exp
import wandb

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-j", default=2, type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--limit", default=16, type=int)
parser.add_argument("--wbid", type=bool, default=False)

args, unknown = parser.parse_known_args()

name = args.name
limit = args.limit
run_info = eval('exp.' + name)

print(unknown)

def parse(it):
    key, value = next(it, (None, None))
    if key is None:
        yield ""
        return
    
    suffix = list(parse(it))

    if isinstance(value, tuple):
        for v in value:
            if isinstance(v, str) and v[0] == '=':
                sep = ""
            else:
                sep = " "
            if isinstance(v, list):
                v = ",".join(map(str, v))
            for s in suffix:
                # yield f"--{key}{sep}{v} ".replace("'", "") + s
                yield f"--{key}{sep}{v} " + s
    else:
        if isinstance(value, str) and value[0] == '=':
            sep = ""
        else:
            sep = " "

        if isinstance(value, list):
            value = ",".join(map(str, value))
        for s in suffix:
            # yield f"--{key}{sep}{value} ".replace("'", "") + s
            yield f"--{key}{sep}{value} " + s



def make_command_list(run_info):
    prefix = f"python3 {run_info['script']} "
    python_command_list = []
    it = iter(run_info['config'].items())
    gen = parse(it)
    # if args.wbid:
    #     gen.append(f" --wbid {wandb.util.generate_id()} ")
    for command in gen:
        if args.wbid:
            command += f" --wbid {wandb.util.generate_id()} "
        command = prefix + command.strip()
        python_command_list.append(command)
    return python_command_list

python_command_list = make_command_list(run_info)
num_jobs = len(python_command_list)

num_arr = (num_jobs - 1) // args.j + 1

print("\n".join(python_command_list))

path = os.getcwd()

d_str = "\n ".join(
    [
        "[{}]='{}'".format(i + 1, command)
        for i, command in enumerate(python_command_list)
    ]
)

sbatch_str = f"""#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time={run_info['time']}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_{run_info['priority']}
#SBATCH --requeue
#SBATCH --array=1-{num_arr}%{limit}
#SBATCH --signal=B:USR1@90

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={args.j}
JOB_N={num_jobs}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
source ~/.bashrc
micromamba activate aorl
export PYTHONPATH="../:${{PYTHONPATH}}"

declare -a commands=(
 {d_str}
)

cd {path}

parallel --delay 20 --linebuffer -j {args.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
"""

with open(f"sbatch/{name}.sh", "w") as f:
    f.write(sbatch_str)