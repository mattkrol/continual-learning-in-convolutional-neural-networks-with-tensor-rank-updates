#!/usr/bin/env python3
#
# This script automates the process of creating a slurm sbatch script.
#
# Author: Matt Krol

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('arguments')
parser.add_argument('-t', type=int, default=0)
args = parser.parse_args()

job_dir = os.path.join(os.getcwd(), 'jobs')
os.makedirs(job_dir, exist_ok=True)

job_log_dir = os.path.join(job_dir, 'logs')
os.makedirs(job_log_dir, exist_ok=True)

job_script_dir = os.path.join(job_dir, 'scripts')
os.makedirs(job_script_dir, exist_ok=True)

DIR = os.path.join(job_log_dir, args.name)

os.makedirs(DIR)

script_file = os.path.join(job_script_dir, '{}.sh'.format(args.name))

PARAMS = {
    'NAME' : args.name,
    'DIR' : DIR,
    'ARGS' : args.arguments
}

template = []
template.append('''\
#!/bin/bash -l
#SBATCH --job-name={NAME}
#SBATCH --output={DIR}/stdout.txt
#SBATCH --error={DIR}/stderr.txt
#SBATCH --mail-user=mrk7339@rit.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --time=5-0:0:0
#SBATCH --account=tensors
#SBATCH --partition=tier3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1

spack load --first python@3.8.3

source venv/bin/activate

python train.py {ARGS}
''')
template.append('''\
#!/bin/bash -l
#SBATCH --job-name={NAME}
#SBATCH --output={DIR}/stdout.txt
#SBATCH --error={DIR}/stderr.txt
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1

source venv/bin/activate

python train.py {ARGS}
''')

with open(script_file, 'w') as f:
    f.write(template[args.t].format(**PARAMS))
