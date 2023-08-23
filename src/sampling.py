#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms

def sample_random_indices(indices, num_samples):
    return np.random.choice(indices, num_samples, replace=False)

def distribute_shards(dict_users, idxs, num_imgs, shard_indices):
    for shard in shard_indices:
        dict_users[i] = np.concatenate(
            (dict_users[i], idxs[shard * num_imgs:(shard + 1) * num_imgs]), axis=0
        )

def emnist_iid(dataset, num_users):
    num_items_per_user = 2400
    dict_users = {i: np.array([]) for i in range(num_users)}
    all_idxs = np.arange(len(dataset))
    
    for i in range(num_users):
        selected_idxs = sample_random_indices(all_idxs, num_items_per_user)
        dict_users[i] = selected_idxs
        all_idxs = np.setdiff1d(all_idxs, selected_idxs)
    
    return dict_users

def emnist_noniid(dataset, num_users, number_of_classes_group1_user, noniidness_end_id, mix):
    num_shards, num_imgs = 1128, 100
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    if mix == 0:
        if number_of_classes_group1_user == 0:
            for i in range(num_users):
                shard_indices = [j for j in range(i * 24, i * 24 + 24)]
                distribute_shards(dict_users, idxs, num_imgs, shard_indices)
        else:
            already_gotten_classes = []
            for i in range(noniidness_end_id):
                rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
                while rand_class in already_gotten_classes:
                    rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
                already_gotten_classes.append(rand_class)
                shard_indices = [rand_class * 24 + j for j in range(24 * number_of_classes_group1_user)]
                distribute_shards(dict_users, idxs, num_imgs, shard_indices)

            num_items = 2400
            all_idxs = np.arange(len(dataset))
            list_of_removed_classes = []

            for j in range(noniidness_end_id):
                list_of_removed_classes.extend(dataset.train_labels[dict_users[j]])

            set_of_removed_classes = set(list_of_removed_classes)
            to_be_deleted = [idd for idd in idxs if labels[idd] in set_of_removed_classes]
            all_idxs = np.setdiff1d(all_idxs, to_be_deleted)

            for i in range(noniidness_end_id, num_users):
                selected_idxs = sample_random_indices(all_idxs, num_items)
                dict_users[i] = selected_idxs
                all_idxs = np.setdiff1d(all_idxs, selected_idxs)
                
    return dict_users

def cifar100_iid(dataset, num_users):
    num_items_per_user = len(dataset) // num_users
    dict_users = {i: np.array([]) for i in range(num_users)}
    all_idxs = np.arange(len(dataset))
    
    for i in range(num_users):
        selected_idxs = sample_random_indices(all_idxs, num_items_per_user)
        dict_users[i] = selected_idxs
        all_idxs = np.setdiff1d(all_idxs, selected_idxs)
    
    return dict_users

def cifar100_noniid(dataset, num_users, number_of_classes_group1_user, noniidness_end_id):
    num_shards, num_imgs = 500, 100
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    already_gotten_classes = []
    for i in range(noniidness_end_id):
        rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
        while rand_class in already_gotten_classes:
            rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
        already_gotten_classes.append(rand_class)
        shard_indices = [rand_class * 5 + j for j in range(5 * number_of_classes_group1_user)]
        distribute_shards(dict_users, idxs, num_imgs, shard_indices)
    
    num_items = 1000
    all_idxs = np.arange(len(dataset))
    list_of_removed_classes = []
    
    for j in range(noniidness_end_id):
        list_of_removed_classes.extend(dataset.targets[dict_users[j]])
    
    set_of_removed_classes = set(list_of_removed_classes)
    to_be_deleted = [idd for idd in idxs if labels[idd] in set_of_removed_classes]
    all_idxs = np.setdiff1d(all_idxs, to_be_deleted)
    
    for i in range(noniidness_end_id, num_users):
        selected_idxs = sample_random_indices(all_idxs, num_items)
        dict_users[i] = selected_idxs
        all_idxs = np.setdiff1d(all_idxs, selected_idxs)
        
    return dict_users