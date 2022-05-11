#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import json
#from fedlab.utils.dataset.slicing import noniid_slicing

def get_new_train_test(dataset_train, dataset_test, args):
    data = {i:[] for i in range(args.num_classes)}
    for i in range(len(dataset_train)):
        data[dataset_train[i][1]].append(dataset_train[i])
    for i in range(len(dataset_test)):
        data[dataset_test[i][1]].append(dataset_test[i])
    data_train = {i: [] for i in range(args.num_classes)}
    data_test = {i: [] for i in range(args.num_classes)}
    number_of_data = len(dataset_test)+len(dataset_train)
    for i in range(args.num_classes):
        index = [k for k in range(len(data[i]))]
        index_train = random.sample(index, int(len(data[i]) * 0.8))
        train_tmp = [data[i][k] for k in index_train]
        data_train[i] = train_tmp
        index_test = list(set(index) - set(index_train))
        test_tmp = [data[i][k] for k in index_test]
        data_test[i] = test_tmp
        print(len(data_train[i]), len(data_test[i]), len(data[i]))
    tmp = []
    for i in range(args.num_classes):
        tmp.extend(data_train[i])
    random.shuffle(tmp)
    dataset_train = tmp
    tmp = []
    for i in range(args.num_classes):
        tmp.extend(data_test[i])
    random.shuffle(tmp)
    dataset_test = tmp
    return dataset_train, dataset_test

def write_dataset(info):
    with open('dataset.txt','w') as f:
        f.write(str(info))


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    random_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #more_data = set(np.random.choice(random_idxs, num_items, replace=False))
        #dict_users[i] = dict_users[i] | more_data
    #write_dataset(dict_users)
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 280
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    random_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #more_data = set(np.random.choice(random_idxs, num_items, replace=False))
        #dict_users[i] = dict_users[i] | more_data
    return dict_users

def noniid_slicing(dataset, num_clients, num_shards):
    # reference fedlab.utils.dataset.slicing.noniid_slicing
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    labels = []
    for i in range(total_sample_nums):
        labels.append(dataset[i][1])
    #labels = np.array(dataset.targets)
    labels = np.array(labels)
    idxs = np.arange(total_sample_nums)
    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    # assign
    idx_shard = [i for i in range(num_shards)]
    random_idxs = [i for i in range(len(dataset))]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 2) * size_of_shards]),
                axis=0)
    return dict_users

def get_noniid(dataset, args):
    data_indices = noniid_slicing(dataset, num_clients=args.num_users, num_shards=4800)
    return data_indices


