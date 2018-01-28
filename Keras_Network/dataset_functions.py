# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:38:38 2018

@author: Stavros
"""

import numpy as np

###################################
######## Dataset Functions ########
###################################

### Format must be configs.npy and labels.npy in the given folder ###

def load_dataset(data_folder='toy_dataset'):
    """Takes a data folder name and returns the dataset, labels and
    labels in one-hot assuming binary classification"""
    confs = np.load(data_folder+'/configs.npy')
    labels = np.load(data_folder+'/labels.npy')
    
    labels_hot = np.empty([len(confs), 2])
    labels_hot[:,0] = (labels == 0).astype(np.int)
    labels_hot[:,1] = (labels == 1).astype(np.int)
    
    return confs, labels, labels_hot

def split_dataset(train_percentage, data_folder='toy_dataset'):
    """Takes a data folder name and returns the dataset and one-hot labels
    for training and testing, splitted according to the percentage"""
    confs, labels, labels_hot = load_dataset(data_folder)
    total_samples = len(confs)
    # Number of samples in training and testing
    train_samples = np.int(train_percentage * total_samples)
    # Randomly choose data for splitting
    indices_list = np.arange(total_samples)
    np.random.shuffle(indices_list)
    # Data for training
    train_confs = confs[indices_list[:train_samples]]
    train_labels = labels_hot[indices_list[:train_samples]]
    # Data for testing
    test_confs = confs[indices_list[train_samples:]]
    test_labels = labels_hot[indices_list[train_samples:]]
    
    return train_confs, train_labels, test_confs, test_labels