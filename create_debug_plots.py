#!/usr/bin/env python3
#
# This file contains a script that will create debug related plots in all of
# the results in the results directory.
#
# Author: Matt Krol

import os
import json
import re
import pickle
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def make_plot(result_path):
    try:
        csv_file = os.path.join(result_path, 'data.csv')
        df = pd.read_csv(csv_file)
    except:
        print('WARNING: The file \'{}\' does not exist.'
              ' Skipping this plot.'.format(csv_file))
        return

    with open(os.path.join(result_path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    title = ''
    if not meta['no_cp']:
        title += 'CP; '
    else:
        title += 'No CP; '
    if meta['no_freeze']:
        title += 'No Freeze; '
    else:
        title += 'Freeze; '
    if meta['separate']:
        title += 'Separate; '

    fig, ax = plt.subplots()

    ax.plot(df['epoch'], df['train_loss_mean'], 'b', label='Mean Train Loss')
    ax.fill_between(df['epoch'], df['train_loss_mean'] + df['train_loss_std'], df['train_loss_mean'] - df['train_loss_std'], color='b', alpha=0.1)

    ax.plot(df['epoch'], df['test_loss_mean'], 'r', label='Mean Test Loss')
    ax.fill_between(df['epoch'], df['test_loss_mean'] + df['test_loss_std'], df['test_loss_mean'] - df['test_loss_std'], color='r', alpha=0.1)

    ax.plot(df['epoch'], df['train_fe_loss_mean'], 'c', label='Mean Train FE')
    ax.plot(df['epoch'], df['test_fe_loss_mean'], 'm', label='Mean Test FE')

    ax.plot(df['epoch'], df['l1_loss'], 'y', label='L1')

    ax.plot(df['epoch'], df['sparsity_mean'], 'g', label='Mean Density')
    ax.fill_between(df['epoch'], df['sparsity_mean'] - df['sparsity_std'], df['sparsity_mean'] + df['sparsity_std'], color='g', alpha=0.1)

    ax.grid()
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.legend()
    #ax.set_ylim((0, 10))
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'loss_debug.png'))
    plt.close()

    fig, ax = plt.subplots()

    ax.plot(df['epoch'], df['train_accuracy'], 'b', label='Train')
    ax.plot(df['epoch'], df['test_accuracy'], 'r', label='Test')

    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_yticks(np.linspace(0, 1, 11))
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'accuracy_debug.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['epoch'], df['train_loss_mean'], 'b', label='Train Loss')
    ax.plot(df['epoch'], df['test_loss_mean'], 'r', label='Test Loss')
    ax.fill_between(df['epoch'], df['train_loss_mean'] + df['train_loss_std'], df['train_loss_mean'] - df['train_loss_std'], color='b', alpha=0.1)
    ax.fill_between(df['epoch'], df['test_loss_mean'] + df['test_loss_std'], df['test_loss_mean'] - df['test_loss_std'], color='r', alpha=0.1)
    ax.grid(axis='x')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    #ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_ylim((0, 5))

    ax1 = ax.twinx()
    ax1.plot(df['epoch'], df['train_accuracy'], 'c', label='Train Accuracy')
    ax1.plot(df['epoch'], df['test_accuracy'], 'm', label='Test Accuracy')
    ax1.grid()
    ax1.set_yticks(np.linspace(0, 1, 11))
    ax1.set_ylim((0, 1.0))
    ax1.set_ylabel('Accuracy')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0, 1, 1, 0), loc='upper left', ncol=2)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'plot.png'))
    plt.close()

    files = os.listdir(os.path.join(result_path, 'confusion_matrices'))
    for file in files:
        if re.fullmatch(r'^task[0-9]+_(test|train)_confusion.pickle$', file):
            fig, ax = plt.subplots(figsize=(6, 5))
            with open(os.path.join(result_path, 'confusion_matrices', file), 'rb') as f:
                cmat = pickle.load(f)
            sns.heatmap(cmat, ax=ax, cmap='hot')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(file)
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, 'confusion_matrices', '{}.png'.format(file)))
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument('--all', metavar='RESULT_ROOT',
                               default='results')
    command_group.add_argument('--single', metavar='RESULT_DIR')
    args = parser.parse_args()

    if args.single is None:
        if not os.path.isdir(os.path.join(os.getcwd(), args.all)):
            raise SystemExit('ERROR: The directory \'{}\''
                             ' does not exist.'.format(args.all))

        dirs = os.listdir(args.all)

        for d in tqdm(dirs, ascii=True, dynamic_ncols=True):
            result_path = os.path.join(os.getcwd(), args.all, d) 

            if not os.path.isdir(result_path):
                continue

            make_plot(result_path)
    else:
        result_path = args.single

        if not os.path.isdir(result_path):
            raise SystemExit('ERROR: The directory \'{}\''
                             ' does not exist.'.format(args.single))

        make_plot(result_path)


if __name__ == '__main__':
    main()
