#!/usr/bin/env python3
#
# This script automatically generates task configurations, slurm sbatch
# scripts, and the slurm log directories.
#
# Author: Matt Krol

import json
import os
import argparse

import numpy as np
from util.dataset import CIFAR100, CIFAR10, MINIIMAGENET, CUB200

MULTITASK_ARGS = "--no-cp --no-freeze --separate --prefix {JOBNAME} --epochs-per-task {EPOCHS} --dataset {DATASETUPPER}"

PARALLEL_ARGS = "--task-config config/{JOBNAME}.json --no-cp --no-freeze --separate --prefix {JOBNAME} --epochs-per-task {EPOCHS} --dataset {DATASETUPPER}"

PARALLEL_LOW_RANK_ARGS = "--task-config config/{JOBNAME}.json --no-increment-rank --no-freeze --separate --prefix {JOBNAME} --epochs-per-task {EPOCHS} --cp-mode 4 --dataset {DATASETUPPER}"

CONTINUAL_ARGS = "--task-config config/{JOBNAME}.json --use-selectors --prefix {JOBNAME} --epochs-per-task {EPOCHS} --cp-mode 4 --dataset {DATASETUPPER}"

template = []
template.append(
    """\
#!/bin/bash -l
#SBATCH --job-name={NAME}
#SBATCH --output={DIR}/stdout.txt
#SBATCH --error={DIR}/stderr.txt
#SBATCH --mail-user=mrk7339@rit.edu
#SBATCH --mail-type=TIME_LIMIT
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

source venv/bin/activate

python train.py {ARGS}
"""
)
template.append(
    """\
#!/bin/bash -l
#SBATCH --job-name={NAME}
#SBATCH --output={DIR}/stdout.txt
#SBATCH --error={DIR}/stderr.txt
#SBATCH --mail-user=mrk7339@rit.edu
#SBATCH --mail-type=NONE
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
"""
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="CIFAR100",
        choices=["CIFAR100", "CIFAR10", "MINIIMAGENET", "CUB200"],
    )
    parser.add_argument("--dataset-location", default="datasets")
    parser.add_argument("--number-of-tasks", default=20, type=int)
    parser.add_argument("--classes-per-task", default=5, type=int)
    parser.add_argument("--realizations", default=10, type=int)
    parser.add_argument("--rank-inc", default=[1], nargs="+", type=int)
    parser.add_argument("--epochs", default=750, type=int)
    parser.add_argument("--script-type", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    rng = np.random.default_rng()
    dataset = args.dataset
    datasets_loc = args.dataset_location
    tasks = args.number_of_tasks
    classes_per_task = args.classes_per_task
    n = args.realizations
    epochs = args.epochs
    rank_inc = args.rank_inc

    os.makedirs("config", exist_ok=True)

    job_dir = os.path.join(os.getcwd(), "jobs")
    os.makedirs(job_dir, exist_ok=True)

    job_log_dir = os.path.join(job_dir, "logs")
    os.makedirs(job_log_dir, exist_ok=True)

    job_script_dir = os.path.join(job_dir, "scripts")
    os.makedirs(job_script_dir, exist_ok=True)

    if dataset == "CIFAR10":
        data_loc = os.path.join(datasets_loc, "cifar-10-batches-py")
        train_data = CIFAR100(data_loc, train=True)
        test_data = CIFAR100(data_loc, train=False)
        prefix = "cifar10"
    elif dataset == "CIFAR100":
        data_loc = os.path.join(datasets_loc, "cifar-100-python")
        train_data = CIFAR100(data_loc, train=True)
        test_data = CIFAR100(data_loc, train=False)
        prefix = "cifar100"
    elif dataset == "MINIIMAGENET":
        data_loc = datasets_loc
        train_data = MINIIMAGENET(data_loc, train=True)
        test_data = MINIIMAGENET(data_loc, train=False)
        prefix = "miniimagenet"
    elif dataset == "CUB200":
        data_loc = os.path.join(datasets_loc, "cub200")
        train_data = CUB200(data_loc, train=True)
        test_data = CUB200(data_loc, train=False)
        prefix = "cub200"
    else:
        raise ValueError("invalid dataset!")

    # Create multitask job script, i.e., the entire dataset.
    job_name = "{}-multitask".format(prefix)
    script_file = os.path.join(job_script_dir, "{}.sh".format(job_name))
    DIR = os.path.join(job_log_dir, job_name)
    os.makedirs(DIR)
    d = {"JOBNAME": job_name, "DATASETUPPER": prefix.upper(), "EPOCHS": epochs}
    ARGS = MULTITASK_ARGS.format(**d)
    PARAMS = {"NAME": job_name, "DIR": DIR, "ARGS": ARGS}
    with open(script_file, "w") as f:
        f.write(template[args.script_type].format(**PARAMS))

    for i in range(n):
        class_labels = np.array(train_data.classes.copy())
        for j, r in enumerate(rank_inc):
            # Generate the continual setup.
            if j == 0:
                task_config = []
                for t in range(tasks):
                    perm = rng.permutation(len(class_labels))[:classes_per_task]
                    task_config.append([list(class_labels[perm]), r])
                    class_labels = np.delete(class_labels, perm)
            else:
                for task in task_config:
                    task[1] = r
            job_name = "{}-c{}-continual-r{}".format(prefix, i + 1, r)
            task_config_loc = os.path.join("config", "{}.json".format(job_name))
            with open(task_config_loc, "w") as f:
                json.dump(task_config, f, indent=4)
            script_file = os.path.join(job_script_dir, "{}.sh".format(job_name))
            DIR = os.path.join(job_log_dir, job_name)
            os.makedirs(DIR)
            d = {
                "JOBNAME": job_name,
                "DATASET": prefix,
                "DATASETUPPER": prefix.upper(),
                "EPOCHS": epochs,
            }
            ARGS = CONTINUAL_ARGS.format(**d)
            PARAMS = {"NAME": job_name, "DIR": DIR, "ARGS": ARGS}
            with open(script_file, "w") as f:
                f.write(template[args.script_type].format(**PARAMS))

            # Generate parallel setup.
            job_name = "{}-c{}-parallel-r{}".format(prefix, i + 1, r)
            task_config_loc = os.path.join("config", "{}.json".format(job_name))
            with open(task_config_loc, "w") as f:
                json.dump(task_config, f, indent=4)
            script_file = os.path.join(job_script_dir, "{}.sh".format(job_name))
            DIR = os.path.join(job_log_dir, job_name)
            os.makedirs(DIR)
            d = {
                "JOBNAME": job_name,
                "DATASET": prefix,
                "DATASETUPPER": prefix.upper(),
                "EPOCHS": epochs,
            }
            ARGS = PARALLEL_LOW_RANK_ARGS.format(**d)
            PARAMS = {"NAME": job_name, "DIR": DIR, "ARGS": ARGS}
            with open(script_file, "w") as f:
                f.write(template[args.script_type].format(**PARAMS))

        # Create the parallel stock job script
        for task in task_config:
            task[1] = 0
        job_name = "{}-c{}-parallel-stock".format(prefix, i + 1)
        task_config_loc = os.path.join("config", "{}.json".format(job_name))
        with open(task_config_loc, "w") as f:
            json.dump(task_config, f, indent=4)
        script_file = os.path.join(job_script_dir, "{}.sh".format(job_name))
        DIR = os.path.join(job_log_dir, job_name)
        os.makedirs(DIR)
        d = {
            "JOBNAME": job_name,
            "DATASET": prefix,
            "DATASETUPPER": prefix.upper(),
            "EPOCHS": epochs,
        }
        ARGS = PARALLEL_ARGS.format(**d)
        PARAMS = {"NAME": job_name, "DIR": DIR, "ARGS": ARGS}
        with open(script_file, "w") as f:
            f.write(template[args.script_type].format(**PARAMS))


if __name__ == "__main__":
    main()
