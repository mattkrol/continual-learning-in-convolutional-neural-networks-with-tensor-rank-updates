# This source code contains all of the dataset classes. For all of the
# datasets, you can filter by class which comes in handy for generating
# different task setups.
#
# Author: Matt Krol

import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def unpickle(file):
    with open(file, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj


class CIFAR100(data.Dataset):
    # Names of the CIFAR100 files.
    filenames = ('test', 'train', 'meta')


    def __init__(self, img_dir, classes=None, train=True,
                 transform=None, shuffle=True):
        # Create transforms.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Get the images.
        data_path = os.path.join(img_dir, CIFAR100.filenames[train])
        data_obj = unpickle(data_path)

        # Reshape the image data into RGB form.
        self.img_data = data_obj[b'data'].reshape(-1, 3, 32, 32)
        self.img_data = self.img_data.transpose((0, 2, 3, 1))

        # Get the data labels.
        self.img_labels = np.array(data_obj[b'fine_labels'], dtype=int)

        # Meta data, i.e., the names of the classes.
        meta_path = os.path.join(img_dir, CIFAR100.filenames[2])
        meta_obj = unpickle(meta_path)

        if classes is None:
            self.classes = [ lbl.decode() for lbl in meta_obj[b'fine_label_names'] ]
        else:
            self.classes = classes

            tmp = [ lbl.decode() for lbl in meta_obj[b'fine_label_names'] ]

            # Original class to index mapping.
            ctoidx0 = { c : i for i, c in enumerate(tmp) }

            # Original index to class mapping.
            idxtoc0 = { i : c for i, c in enumerate(tmp) }

            # New class to index mapping.
            ctoidx1 = { c : i for i, c in enumerate(self.classes) }

            mask = np.zeros(len(self.img_labels), dtype=bool)

            # Create mask to filter out unwanted classes.
            for c in self.classes:
                mask = mask | (self.img_labels == ctoidx0[c])

            # Apply the mask to the labels and image set to filter.
            self.img_labels = self.img_labels[mask]
            self.img_data = self.img_data[mask,...]

            # Change labels to be from 0 to C - 1.
            for i in range(self.img_labels.size):
                c = idxtoc0[self.img_labels[i]]
                self.img_labels[i] = ctoidx1[c]

        if shuffle:
            # Shuffle up the data.
            perm = np.random.permutation(self.img_data.shape[0])
            self.img_data = self.img_data[perm,...]
            self.img_labels = self.img_labels[perm]

        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = Image.fromarray(self.img_data[idx,...])
        image = self.transform(image)
        return image, label


class CIFAR10(data.Dataset):
    filenames = (('test_batch',), ('data_batch_1', 'data_batch_2',
                  'data_batch_3', 'data_batch_4', 'data_batch_5'))


    def __init__(self, img_dir, classes=None, train=True,
                 transform=None, shuffle=True):
        # Create transforms.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.img_labels = []
        self.img_data = []

        # Unpack the images in the dataset.
        for f in CIFAR10.filenames[train]:
            data_obj = unpickle(os.path.join(img_dir, f))
            self.img_data.append(data_obj[b'data'])
            self.img_labels.extend(data_obj[b'labels'])

        # Reshape images into RGB form.
        self.img_data = np.vstack(self.img_data).reshape(-1, 3, 32, 32)
        self.img_data = self.img_data.transpose((0, 2, 3, 1))

        self.img_labels = np.array(self.img_labels, dtype=int)

        meta_path = os.path.join(img_dir, 'batches.meta')
        meta_obj = unpickle(meta_path)

        if classes is None:
            self.classes = [ lbl.decode() for lbl in meta_obj[b'fine_label_names'] ]
        else:
            self.classes = classes

            tmp = [ lbl.decode() for lbl in meta_obj[b'label_names'] ]

            # Original class to index mapping.
            ctoidx0 = { c : i for i, c in enumerate(tmp) }

            # Original index to class mapping.
            idxtoc0 = { i : c for i, c in enumerate(tmp) }

            # New class to index mapping.
            ctoidx1 = { c : i for i, c in enumerate(self.classes) }

            mask = np.zeros(len(self.img_labels), dtype=bool)

            # Create mask to filter out unwanted classes.
            for c in self.classes:
                mask = mask | (self.img_labels == ctoidx0[c])

            # Apply the mask to the labels and image set to filter.
            self.img_labels = self.img_labels[mask]
            self.img_data = self.img_data[mask,...]

            # Change labels to be from 0 to C - 1.
            for i in range(self.img_labels.size):
                c = idxtoc0[self.img_labels[i]]
                self.img_labels[i] = ctoidx1[c]

        if shuffle:
            # Shuffle up the data.
            perm = np.random.permutation(self.img_data.shape[0])
            self.img_data = self.img_data[perm,...]
            self.img_labels = self.img_labels[perm]

        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = Image.fromarray(self.img_data[idx,...])
        image = self.transform(image)
        return image, label


class MINIIMAGENET(data.Dataset):
    filename = 'miniimagenet.pickle'


    def __init__(self, img_dir, classes=None, train=True,
                 transform=None, shuffle=True):
        # Create transforms.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        with open(os.path.join(img_dir, MINIIMAGENET.filename), 'rb') as f:
            data = pickle.load(f)

        # Create test and train splits here.
        sl = slice(0, 500) if train else slice(500, None)
        filtered_data = []
        filtered_labels = []
        for l in range(100):
            mask = data['labels'] == l
            filtered_labels.append(data['labels'][mask,...][sl,...])
            filtered_data.append(data['images'][mask,...][sl,...])
        self.img_labels = np.concatenate(filtered_labels, dtype=int)
        self.img_data = np.concatenate(filtered_data, axis=0, dtype=np.uint8)

        if classes is None:
            self.classes = [ str(i) for i in range(100) ]
        else:
            self.classes = classes
            tmp = [ str(i) for i in range(100) ]

            # Original class to index mapping.
            ctoidx0 = { c : i for i, c in enumerate(tmp) }

            # Original index to class mapping.
            idxtoc0 = { i : c for i, c in enumerate(tmp) }

            # New class to index mapping.
            ctoidx1 = { c : i for i, c in enumerate(self.classes) }

            mask = np.zeros(len(self.img_labels), dtype=bool)

            # Create mask to filter out unwanted classes.
            for c in self.classes:
                mask = mask | (self.img_labels == ctoidx0[c])

            # Apply the mask to the labels and image set to filter.
            self.img_labels = self.img_labels[mask]
            self.img_data = self.img_data[mask,...]

            # Change labels to be from 0 to C - 1.
            for i in range(self.img_labels.size):
                c = idxtoc0[self.img_labels[i]]
                self.img_labels[i] = ctoidx1[c]

        if shuffle:
            # Shuffle up the data.
            perm = np.random.permutation(self.img_data.shape[0])
            self.img_data = self.img_data[perm,...]
            self.img_labels = self.img_labels[perm]

        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = Image.fromarray(self.img_data[idx,...])
        image = self.transform(image)
        return image, label


class CUB200(data.Dataset):
    def __init__(self, img_dir, classes=None, train=True,
                 transform=None, shuffle=True):
        # Create transforms.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.base_dir = os.path.join(img_dir, 'CUB_200_2011')

        self.img_locs = []
        self.img_labels = []

        # Read in the image locations and the labels.
        with open(os.path.join(self.base_dir, 'images.txt'), 'r') as f1,\
             open(os.path.join(self.base_dir, 'image_class_labels.txt'), 'r') as f2:
            for line1, line2 in zip(f1.readlines(), f2.readlines()):
                fpath = line1.split()[-1]
                label = line2.split()[-1]
                self.img_locs.append(fpath)
                self.img_labels.append(int(label))
        self.img_locs = np.array(self.img_locs, dtype=str)
        self.img_labels = np.array(self.img_labels, dtype=int) - 1

        # Get the train test split information.
        train_mask = []
        with open(os.path.join(self.base_dir, 'train_test_split.txt'), 'r') as f:
            for line in f.readlines():
                train_mask.append(int(line.split()[-1]))
        train_mask = np.array(train_mask, dtype=bool)

        # Use the mask to get the test or train set.
        if train:
            self.img_locs = self.img_locs[train_mask]
            self.img_labels = self.img_labels[train_mask]
        else:
            self.img_locs = self.img_locs[~train_mask]
            self.img_labels = self.img_labels[~train_mask]

        # Get the class names.
        tmp = []
        with open(os.path.join(self.base_dir, 'classes.txt'), 'r') as f:
            for line in f.readlines():
                tmp.append(line.split()[-1][4:])

        if classes is None:
            self.classes = tmp
        else:
            self.classes = classes

            # Original class to index mapping.
            ctoidx0 = { c : i for i, c in enumerate(tmp) }

            # Original index to class mapping.
            idxtoc0 = { i : c for i, c in enumerate(tmp) }

            # New class to index mapping.
            ctoidx1 = { c : i for i, c in enumerate(self.classes) }

            mask = np.zeros(len(self.img_labels), dtype=bool)

            # Create mask to filter out unwanted classes.
            for c in self.classes:
                mask = mask | (self.img_labels == ctoidx0[c])

            # Apply the mask to the labels and image set to filter.
            self.img_labels = self.img_labels[mask]
            self.img_locs = self.img_locs[mask,...]

            # Change labels to be from 0 to C - 1.
            for i in range(self.img_labels.size):
                c = idxtoc0[self.img_labels[i]]
                self.img_labels[i] = ctoidx1[c]

        if shuffle:
            # Shuffle up the data.
            perm = np.random.permutation(self.img_locs.size)
            self.img_locs = self.img_locs[perm]
            self.img_labels = self.img_labels[perm]

        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = Image.open(os.path.join(self.base_dir, 'images', self.img_locs[idx]))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, label
