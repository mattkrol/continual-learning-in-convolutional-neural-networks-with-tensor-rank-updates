# Source code that creates a CSV to store different data while training.
#
# Author: Matt Krol

import os
import csv


class Results(object):
    fieldnames = (
        'epoch',
        'lr',
        'parameters',
        'itr_time',
        'train_batch_loss_mean',
        'train_batch_loss_std',
        'train_time',
        'train_loss_mean',
        'train_loss_std',
        'train_accuracy',
        'train_fe_loss_mean',
        'train_fe_loss_std',
        'test_loss_mean',
        'test_loss_std',
        'test_accuracy',
        'test_fe_loss_mean',
        'test_fe_loss_std',
        'l1_loss',
        'sparsity_mean',
        'sparsity_std'
    )


    def __init__(self, results_dir):
        self.results_dir = results_dir

        self.csv_file = os.path.join(self.results_dir, 'data.csv')

        with open(self.csv_file, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=Results.fieldnames)
            csv_writer.writeheader()


    def append(self, epoch, lr, parameters, itr_time,
               train_batch_loss_mean, train_batch_loss_std, train_time,
               train_loss_mean, train_loss_std, train_accuracy,
               train_fe_loss_mean, train_fe_loss_std,
               test_loss_mean, test_loss_std, test_accuracy,
               test_fe_loss_mean, test_fe_loss_std,
               l1_loss, sparsity_mean, sparsity_std):
        test_data = {
            'epoch' : epoch,
            'lr' : lr,
            'parameters' : parameters,
            'itr_time' : itr_time,
            'train_batch_loss_mean' : train_batch_loss_mean,
            'train_batch_loss_std' : train_batch_loss_std,
            'train_time' : train_time,
            'train_loss_mean' : train_loss_mean,
            'train_loss_std' : train_loss_std,
            'train_accuracy' : train_accuracy,
            'train_fe_loss_mean' : train_fe_loss_mean,
            'train_fe_loss_std' : train_fe_loss_std,
            'test_loss_mean' : test_loss_mean,
            'test_loss_std' : test_loss_std,
            'test_accuracy' : test_accuracy,
            'test_fe_loss_mean' : test_fe_loss_mean,
            'test_fe_loss_std' : test_fe_loss_std,
            'l1_loss' : l1_loss,
            'sparsity_mean' : sparsity_mean,
            'sparsity_std' : sparsity_std
        }

        for field in Results.fieldnames:
            if field not in test_data.keys():
                test_data[field] = None

        with open(self.csv_file, 'a+') as f:
            csv_writer = csv.DictWriter(f, fieldnames=Results.fieldnames)
            csv_writer.writerow(test_data)
