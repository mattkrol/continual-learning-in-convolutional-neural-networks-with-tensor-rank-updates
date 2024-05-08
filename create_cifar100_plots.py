#!/usr/bin/env python3
#
# This script creates the plots for the paper.
#
# Author: Matt Krol

import os
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
import torch

from resnet import ResNet18

rc("text", usetex=True)
rc("font", size=15)
rc("font", family="serif")
rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}")


def get_params(base_dir, num_tasks, num_classes):
    with open(os.path.join(base_dir, "ranks.pickle"), "rb") as f:
        ranks = pickle.load(f)
    params = np.zeros(num_tasks, dtype=int)
    for i in range(num_tasks):
        for t in range(1, i + 2):
            sd = torch.load(os.path.join(base_dir, "state_dicts", f"sd_task{t}_final.pt"))
            model = ResNet18(
                num_classes,
                rank=sum(ranks[:t]),
                last_rank=sum(ranks[: t - 1]) if t > 1 else 0,
                mode=4,
            )
            model.load_state_dict(sd)
            if t < i + 1:
                for k, p in model.named_parameters():
                    if re.fullmatch(r"^.*conv[0-9]+\.filters\.[0-9]+\.s$", k):
                        params[i] += p.numel()
                    elif re.fullmatch(r"^fc.(weight|bias)$", k):
                        params[i] += p.numel()
                    elif re.fullmatch(r"^.*bn[0-9]+\.(weight|bias)$", k):
                        params[i] += p.numel()
            else:  # t == num_tasks
                params[i] += sum(p.numel() for p in model.parameters())
    return params


def main():
    n = 10  # number of realizations
    e = 750  # number of epochs per task
    t = 20  # tasks
    m = 5  # classes per task

    plot_dir = os.path.join("plots", "cifar100")
    os.makedirs(plot_dir, exist_ok=True)

    base_dir = "results"

    ranks = [1, 5, 10, 15, 20]
    dataset = "cifar100"
    experiments = (
        [f"continual-r{r}" for r in ranks]
        + [f"parallel-r{r}" for r in ranks]
        + ["parallel-stock", "continual-r5plus1"]
    )

    if not all(
        [os.path.exists(os.path.join(plot_dir, f)) for f in [f"{dataset}_test_acc.pkl", f"{dataset}_params.pkl"]]
    ):
        test_acc = np.empty((len(experiments), t, n), dtype=np.double)
        params = np.empty((len(experiments), t, n), dtype=int)
        for i, suffix in enumerate(experiments):
            for j, c in enumerate(range(1, n + 1)):
                try:
                    exp_dir = os.path.join(base_dir, f"{dataset}-c{c}-{suffix}")
                    df = pd.read_csv(os.path.join(exp_dir, "data.csv"))
                    test_acc[i, :, j] = df["test_accuracy"].iloc[e - 1 :: e].to_numpy()
                    if "continual" in suffix:
                        params[i, :, j] = get_params(exp_dir, t, m)
                    else:
                        params[i, :, j] = df["parameters"].iloc[e - 1 :: e].to_numpy()
                except Exception as e:
                    print(exp_dir)
                    raise e
        with open(os.path.join(plot_dir, f"{dataset}_test_acc.pkl"), "wb") as f:
            pickle.dump(test_acc, f)
        with open(os.path.join(plot_dir, f"{dataset}_params.pkl"), "wb") as f:
            pickle.dump(params, f)
    else:
        with open(os.path.join(plot_dir, f"{dataset}_test_acc.pkl"), "rb") as f:
            test_acc = pickle.load(f)
        with open(os.path.join(plot_dir, f"{dataset}_params.pkl"), "rb") as f:
            params = pickle.load(f)

    # Make bar plot figure.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, layout="constrained")

    y = np.mean(test_acc, axis=(1, 2))

    width = 0.25
    br = np.arange(3)
    br1 = [x - width / 2 for x in br]
    br2 = [x + width / 2 for x in br]

    ax[0].barh(br1, y[:3] * 100, height=width, color="g", label="Continual")
    ax[0].barh(br2, y[5:8] * 100, height=width, color="b", label="Parallel Low-Rank")
    ax[0].axvline(y[10] * 100, color="r", label="Parallel ResNet18")
    ax[0].axvline(y[11] * 100, color="m", linestyle="-", label="Continual 5plus1")

    ax[0].set_yticks([r for r in range(3)], ranks[:3])
    ax[0].set_ylabel("Rank")
    ax[0].set_xlabel("Task Accuracy")
    ax[0].set_xticks(np.linspace(0, 100, 11))
    ax[0].set_axisbelow(True)
    ax[0].grid(axis="x", alpha=0.3)

    y = np.max(params, axis=(1, 2))

    ax[1].barh(br1, y[:3], height=width, color="g")
    ax[1].barh(br2, y[5:8] * t, height=width, color="b")
    ax[1].axvline(y[10] * t, color="r")
    ax[1].axvline(y[11], color="m", linestyle="-")

    ax[1].set_yticks([r for r in range(3)], ranks[:3])
    ax[1].set_xlabel("Parameters")
    ax[1].set_xscale("log")
    ax[1].set_axisbelow(True)
    ax[1].grid(axis="x", alpha=0.3, which="both")

    fig.legend(framealpha=1.0)
    fig.align_labels()
    plt.savefig(os.path.join(plot_dir, f"{dataset}_fig1.png"))
    # plt.show()

    # Make a CSV of the data.
    amean = np.mean(test_acc, axis=(1, 2))
    astd = np.std(test_acc, axis=(1, 2))
    df = pd.DataFrame(
        {
            "Method": [
                "Continual Rank 1",
                "Continual Rank 5",
                "Continual Rank 10",
                "Parallel Rank 1",
                "Parallel Rank 5",
                "Parallel Rank 10",
                "Parallel ResNet18",
                "Continual 5plus1",
            ],
            "Task Accuracy Mean": np.concatenate([amean[:3], amean[5:8], [amean[10]], [amean[11]]]),
            "Task Accuracy Standard Deviation": np.concatenate([astd[:3], astd[5:8], [astd[10]], [astd[11]]]),
            "Final Task Parameters": np.concatenate([y[:3], y[5:8] * t, [y[10] * t], [y[11]]]),
        }
    )
    df.to_csv(os.path.join(plot_dir, f"{dataset}_table1.csv"), index=False)

    # Make line plot figure for each task.
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["b", "g", "r", "c", "m", "y"]

    for i in [0, 1, 2]:
        ax.plot(
            np.arange(t) + 1,
            np.mean(test_acc[i, :, :] * 100, axis=1),
            f"{colors[i]}-",
            label=f"Continual Rank {ranks[i]}",
        )

    for i in [5, 6, 7]:
        ax.plot(
            np.arange(t) + 1,
            np.mean(test_acc[i, :, :] * 100, axis=1),
            f"{colors[i-5]}--",
            label=f"Parallel Rank {ranks[i-5]}",
        )

    ax.plot(
        np.arange(t) + 1,
        np.mean(test_acc[10, :, :] * 100, axis=1),
        colors[-1],
        label="Parallel ResNet18",
    )

    ax.plot(
        np.arange(t) + 1,
        np.mean(test_acc[11, :, :] * 100, axis=1),
        colors[-2],
        label="Continual 5plus1",
    )

    ax.set_yticks(np.arange(5, 100, 5))
    ax.set_xticks(np.arange(t) + 1)
    ax.set_xlabel("Task")
    ax.set_ylabel("Task Accuracy")
    ax.set_ylim([65, 95])
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset}_fig2.png"))

    # Make plot of parameters.
    fig, ax = plt.subplots(figsize=(12, 8))

    for i in [0, 1, 2]:
        ax.plot(
            np.arange(t) + 1, np.mean(params[i, :, :], axis=1), f"{colors[i]}-", label=f"Continual Rank {ranks[i]}"
        )

    for i in [5, 6, 7]:
        ax.plot(
            np.arange(t) + 1,
            np.cumsum(np.mean(params[i, :, :], axis=1)),
            f"{colors[i-5]}--",
            label=f"Parallel Rank {ranks[i-5]}",
        )

    ax.plot(np.arange(t) + 1, np.cumsum(np.mean(params[10, :, :], axis=1)), colors[-1], label="Parallel ResNet18")
    ax.plot(np.arange(t) + 1, np.mean(params[11, :, :], axis=1), colors[-2], label="Continual 5plus1")

    ax.set_xticks(np.arange(t) + 1)
    ax.set_yscale("log")
    ax.set_xlabel("Task")
    ax.set_ylabel("Parameters")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset}_fig3.png"))
    # plt.show()


if __name__ == "__main__":
    main()
