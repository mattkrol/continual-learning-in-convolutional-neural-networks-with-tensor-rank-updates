# This source code contains the task iterator that manages the NN
# initializations and parameter copying for each task.
#
# Author: Matt Krol

import json
import re
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from util.dataset import CIFAR100, CIFAR10, MINIIMAGENET, CUB200
from resnet import ResNet18


torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Tasks(object):
    def __init__(self, args, train_kwargs, test_kwargs, device):
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
        self.device = device
        self.index = 0
        self.args = args

        if self.args.task_config is not None:
            # Use config file to configure tasks.
            with open(self.args.task_config, 'r') as f:
                self.task_data = json.load(f)
        else:
            # The empty list means use the whole dataset.
            self.task_data = [[[],1]]

        # The number of tasks.
        self.size = len(self.task_data)

        # Store a history of task ranks.
        self.ranks = []


    def __iter__(self):
        return self


    def __len__(self):
        return self.size


    def __next__(self):
        if self.index < self.size:
            if len(self.task_data[self.index][0]) > 0:
                classes = self.task_data[self.index][0]
            else:
                classes = None

            # Keep a history of ranks.
            self.ranks.append(self.task_data[self.index][1])
            # Use sum of ranks or just use current rank.
            if not self.args.no_increment_rank:
                self.rank = sum(self.ranks)
            else:
                self.rank = self.ranks[-1]

            if self.index < 1:
                last_rank =  0
            else:
                last_rank = sum(self.ranks[:self.index])

            print('Last Rank = {}'.format(last_rank))

            print('Current Rank = {}'.format(self.rank))

            # Determine the dataset.
            if self.args.dataset == 'CIFAR100':
                stats = ((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                dataset_loc = os.path.join(self.args.dataset_location, 'cifar-100-python')
                train_data = CIFAR100(dataset_loc, classes=classes,
                                      train=True, transform=train_transforms)

                test_data = CIFAR100(dataset_loc, classes=classes,
                                     train=False, transform=test_transforms)
            elif self.args.dataset == 'CIFAR10':
                stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                # Setup the train and test loaders.
                dataset_loc = os.path.join(self.args.dataset_location, 'cifar10-batches-py')
                train_data = CIFAR10(dataset_loc, classes=classes,
                                     train=True, transform=train_transforms)

                test_data = CIFAR10(dataset_loc, classes=classes,
                                    train=False, transform=test_transforms)
            elif self.args.dataset == 'MINIIMAGENET':
                stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(84, padding=10),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                # Setup the train and test loaders.
                dataset_loc = self.args.dataset_location
                train_data = MINIIMAGENET(dataset_loc, classes=classes,
                                          train=True, transform=train_transforms)

                test_data = MINIIMAGENET(dataset_loc, classes=classes,
                                         train=False, transform=test_transforms)
            elif self.args.dataset == 'CUB200': 
                img_size = 224
                stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                train_transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])

                test_transforms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)
                ])
                # Setup the train and test loaders.
                dataset_loc = os.path.join(self.args.dataset_location, 'cub200')
                train_data = CUB200(dataset_loc, classes=classes,
                                    train=True, transform=train_transforms)

                test_data = CUB200(dataset_loc, classes=classes,
                                   train=False, transform=test_transforms)
            else:
                raise ValueError('invalid dataset!')

            train_loader = torch.utils.data.DataLoader(train_data,
                                                       **self.train_kwargs)

            test_loader = torch.utils.data.DataLoader(test_data,
                                                      **self.test_kwargs)

            if self.index == 0 or self.args.separate:
                # First task gets a fresh model always. If the separate flag
                # has been passed every task will get a fresh model.
                self.model = ResNet18(
                    train_data.num_classes,
                    rank=self.rank if not self.args.no_cp else 0,
                    last_rank=last_rank if self.args.use_selectors else 0,
                    mode=self.args.cp_mode
                ).to(self.device)
            else:
                # Next task model update routine.
                last_sd = self.model.state_dict()
                del self.model
                del self.optimizer
                del self.scheduler
                self.model = ResNet18(
                    train_data.num_classes,
                    rank=self.rank if not self.args.no_cp else 0,
                    last_rank=last_rank if self.args.use_selectors else 0,
                    mode=self.args.cp_mode
                ).to(self.device)

                # Copy over parameters from previous task where it makes sense.
                sd = self.model.state_dict()
                for k in sd.keys():
                    # There are no selectors to copy over in this case.
                    if re.fullmatch(r'^.*conv[0-9]+\.filters\.[0-9]+\.s$', k) and self.index <= 1:
                        continue
                    # Copy over old parameters.
                    if sd[k].shape != last_sd[k].shape:
                        if re.fullmatch(r'^.*conv[0-9]+\.filters\.[0-9]+\.factors\.[0-3]+$', k):
                            sd[k][:,:last_sd[k].shape[1]] = last_sd[k]
                        elif re.fullmatch(r'^.*conv[0-9]+\.filters\.[0-9]+\.s$', k):
                            sd[k][:last_sd[k].shape[0]] = last_sd[k]
                        else:
                            raise SystemExit('Did not account for updating key \'{}\'.'.format(k))
                    else:
                        sd[k] = last_sd[k]
                self.model.load_state_dict(sd)

            # Always get a fresh optimizer and scheduler instance at each task.
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.args.lr)

            self.scheduler = StepLR(self.optimizer,
                                    step_size=max(1, int(self.args.lr_step_ratio*self.args.epochs_per_task + 0.5)),
                                    gamma=self.args.lr_reduction_ratio)

            self.index += 1

            tdata = (self.model, self.optimizer, self.scheduler, train_loader,
                     test_loader, last_rank)

            return tdata
        else:
            raise StopIteration
