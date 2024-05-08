#!/usr/bin/env python3
#
# This script trains the NN! See the argument parser for all of the options or
# type 'python train.py --help'.
#
# Author: Matt Krol

import argparse
import time
import datetime
import pickle
import re
import os
import json
import uuid
import random

import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

from util.metrics import ConfusionMatrix
from util.results import Results
from tasks import Tasks


# Fix the seeds.
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# Fix the CUDA worker seeds.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def percent_sparsity(s, p):
    arrs = np.sort(np.abs(s))[::-1]
    cmp = p*np.sum(arrs)
    csum = np.cumsum(arrs)
    return np.argmax(csum >= cmp) + 1


def train(args, model, device, train_loader, optimizer, epoch, task, last_rank):
    model.train()
    train_loss = []
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Reset the gradients to zero.
        optimizer.zero_grad()
        # Pass the training batch through the model.
        output = model(data)
        # Calculate the loss.
        fe_loss = F.cross_entropy(output, target)
        # Add L1 regularization to selector weights.
        l1_loss = 0
        # Calculate the L1 regularization term.
        if not args.no_cp and args.use_selectors and args.lambd > 0.0 and last_rank > 0:
            c = 0
            named_layers = dict(model.named_modules())
            for k in named_layers.keys():
                if named_layers[k].__class__.__name__ == 'CPFilter' and hasattr(named_layers[k], 's'):
                    l1_loss = l1_loss + torch.norm(named_layers[k].s, p=1)
                    c += 1
            l1_loss /= c
            total_loss = (1 - args.lambd)*fe_loss + args.lambd*l1_loss
        else:
            total_loss = fe_loss
        train_loss.append(total_loss.item())
        # Backward compute the gradients.
        total_loss.backward()
        # Freeze last task weights if needed.
        if task != 0 and not args.no_freeze and not args.no_cp:
            with torch.no_grad():
                named_layers = dict(model.named_modules())
                for k in named_layers.keys():
                    if named_layers[k].__class__.__name__ == 'CPFilter':
                        for factor in named_layers[k].factors:
                            factor.grad[:,:last_rank] = 0.
        # Update the parameters of the model.
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch {} | {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                  epoch, (batch_idx + 1)*args.batch_size, len(train_loader.dataset),
                  100*(batch_idx + 1)/len(train_loader), total_loss.item()))
        if args.dry_run:
            break

    # Calculate mini batch statistics.
    train_loss_mean = np.mean(train_loss)
    train_loss_std = np.std(train_loss)

    end = time.time()
    train_time = end - start

    print('Train Epoch {} Completed | Avg Loss: {:.6f} | Time: {}\n'.format(
          epoch, train_loss_mean, str(datetime.timedelta(seconds=round(train_time)))))

    return train_loss_mean, train_loss_std, train_time


def test(args, model, device, test_loader, last_rank, msg='Test Set'):
    model.eval()

    fe_loss = []
    confusion = ConfusionMatrix(device, classes=test_loader.dataset.num_classes)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            fe_loss.extend(F.cross_entropy(output, target, reduction='none').tolist())

            pred = output.argmax(dim=1, keepdim=True)
            confusion.add(pred.flatten(), target.flatten())

        fe_loss = np.array(fe_loss)

        if not args.no_cp and args.use_selectors and last_rank > 0:
            l1_loss = []
            sparsity = []
            named_layers = dict(model.named_modules())
            for k in named_layers.keys():
                if named_layers[k].__class__.__name__ == 'CPFilter' and hasattr(named_layers[k], 's'):
                    l1_loss.append(torch.norm(named_layers[k].s, p=1).item())
                    sparsity.append(percent_sparsity(named_layers[k].s.cpu().numpy(), 0.9))
            l1_loss = np.mean(l1_loss)
            #l1_loss = np.sum(l1_loss)
            sparsity_mean = np.mean(sparsity)
            sparsity_std = np.std(sparsity)
            test_loss = (1 - args.lambd)*fe_loss + args.lambd*l1_loss
        else:
            l1_loss = 0
            sparsity_mean = None
            sparsity_std = None
            test_loss = fe_loss

    fe_loss_mean = np.mean(fe_loss)
    fe_loss_std = np.std(fe_loss)
    
    test_loss_mean = np.mean(test_loss)
    test_loss_std = np.std(test_loss)

    test_accuracy = confusion.accuracy()

    print('{} | Avg Loss: {:.6f} | Accuracy: {:.2f}%\n'.format(
          msg, test_loss_mean, test_accuracy*100))

    return (test_loss_mean, test_loss_std, test_accuracy, confusion.get(),
            fe_loss_mean, fe_loss_std, l1_loss, sparsity_mean, sparsity_std)


def save_model(model, state_dicts_dir, task, name):
    state_dict_file = os.path.join(
        state_dicts_dir,
        'sd_task{}_{}.pt'.format(task + 1, name)
    )
    torch.save(model.state_dict(), state_dict_file)


def save_confusion(test_confusion, train_confusion, confusion_dir, task):
    with open(os.path.join(confusion_dir, 'task{}_test_confusion.pickle'.format(task + 1)), 'wb') as f:
        pickle.dump(test_confusion, f)
    with open(os.path.join(confusion_dir, 'task{}_train_confusion.pickle'.format(task + 1)), 'wb') as f:
        pickle.dump(train_confusion, f)


def save_ranks(results_dir, ranks):
    with open(os.path.join(results_dir, 'ranks.pickle'), 'wb') as f:
        pickle.dump(ranks, f)


def main():
    parser = argparse.ArgumentParser(description='ResNet18 CP ITL')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1250, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs-per-task', type=int, default=300, metavar='N',
                        help='number of epochs to train per task')
    parser.add_argument('--no-cp', action='store_true', default=False,
                        help='do not use cp factorization')
    parser.add_argument('--cp-mode', type=int, choices=[3, 4], default=4, metavar='N',
                        help='mode of the cp decomposition')
    parser.add_argument('--no-freeze', action='store_true', default=False,
                        help='do not freeze tensor train last task parameters')
    parser.add_argument('--separate', action='store_true', default=False,
                        help='use new model instance at each task')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='F',
                        help='learning rate')
    parser.add_argument('--lambd', type=float, default=0.0, metavar='F',
                        help='regularization hyperparameter')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', 'MINIIMAGENET', 'CUB200'],
                        metavar='LOC', help='dataset to use')
    parser.add_argument('--dataset-location', default='datasets')
    parser.add_argument('--task-config', default=None, metavar='LOC',
                        help='task config file location')
    parser.add_argument('--prefix', default=None,
                        help='prefix for directory')
    parser.add_argument('--no-increment-rank', action='store_true', default=False,
                        help='do not increment the rank, use fixed rank')
    parser.add_argument('--use-selectors', action='store_true', default=False,
                        help='use selectors in the CP decomposition')
    parser.add_argument('--lr-step-ratio', type=float, default=1.0,
                        help='step size for learning rate reduction in terms of epochs per task')
    parser.add_argument('--lr-reduction-ratio', type=float, default=0.3,
                        help='learning rate reduction ratio')
    args = parser.parse_args()

    # Create the results directory.
    if args.prefix is not None:
        rtmp = args.prefix
    else:
        rtmp = uuid.uuid4().hex

    results_dir = os.path.join(os.getcwd(), 'results', rtmp)
    os.makedirs(results_dir)

    # Create directory to hold state dicts.
    state_dicts_dir = os.path.join(results_dir, 'state_dicts')
    os.makedirs(state_dicts_dir)

    # Create directory to hold confusion matrices.
    confusion_dir = os.path.join(results_dir, 'confusion_matrices')
    os.makedirs(confusion_dir)

    # Save tasks config for reference.
    if args.task_config is not None:
        with open(args.task_config, 'r') as f:
            task_config = json.load(f)

        with open(os.path.join(results_dir, 'tasks.json'), 'w') as f:
            json.dump(task_config, f, indent=2)

    # Meta data from test for reference.
    meta = {
        'time' : datetime.datetime.now().isoformat(),
    }

    meta.update(args.__dict__)

    with open(os.path.join(results_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('Using device: {}\n'.format('cuda' if use_cuda else 'cpu'))

    train_kwargs = { 'batch_size': args.batch_size }
    test_kwargs = { 'batch_size': args.test_batch_size }

    if use_cuda:
        cuda_kwargs = {'worker_init_fn' : seed_worker,
                       'num_workers' : 2,
                       'generator' : g,
                       'pin_memory' : False,
                       'shuffle' : True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Create the results files.
    results = Results(results_dir)

    epoch_offset = 0

    task_iterator = Tasks(args, train_kwargs, test_kwargs, device)

    for task, tdata in enumerate(task_iterator):
        model, optimizer, scheduler, train_loader, test_loader, last_rank = tdata

        parameters = sum(p.numel() for p in model.parameters())

        # Save initial model. This is mostly for debug.
        #save_model(model, state_dicts_dir, task, 'init')

        for epoch in range(1, args.epochs_per_task + 1):
            start = time.time()

            train_batch_loss_mean, train_batch_loss_std, train_time = train(args, model, device, train_loader, optimizer, epoch + epoch_offset, task, last_rank)

            test_loss_mean, test_loss_std, test_accuracy, test_confusion, test_fe_loss_mean, test_fe_loss_std, _, _, _ = test(args, model, device, test_loader, last_rank)

            train_loss_mean, train_loss_std, train_accuracy, train_confusion, train_fe_loss_mean, train_fe_loss_std, l1_loss, sparsity_mean, sparsity_std = test(args, model, device, train_loader, last_rank, msg='Train Set')

            lr = scheduler.optimizer.param_groups[0]['lr']

            scheduler.step()

            end = time.time()
            itr_time = end - start

            results.append(epoch + epoch_offset, lr, parameters, itr_time,
                           train_batch_loss_mean, train_batch_loss_std, train_time,
                           train_loss_mean, train_loss_std, train_accuracy,
                           train_fe_loss_mean, train_fe_loss_std,
                           test_loss_mean, test_loss_std, test_accuracy,
                           test_fe_loss_mean, test_fe_loss_std,
                           l1_loss, sparsity_mean, sparsity_std)

        # Save final model.
        save_model(model, state_dicts_dir, task, 'final')

        # Save final confusion matrices.
        save_confusion(test_confusion, train_confusion, confusion_dir, task)

        epoch_offset += args.epochs_per_task

    # Save rank history for TTD.
    save_ranks(results_dir, task_iterator.ranks)


if __name__ == '__main__':
    main()
