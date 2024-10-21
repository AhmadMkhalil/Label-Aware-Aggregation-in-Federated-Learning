#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms

def sample_random_indices(indices, num_samples):
    """Samples random indices from a given set of indices.
    
    Args:
    - indices: An array of available indices.
    - num_samples: The number of indices to sample.
    
    Returns:
    - An array of randomly selected indices.
    """
    return np.random.choice(indices, num_samples, replace=False)  # Sample without replacement

def distribute_shards(dict_users, idxs, num_imgs, shard_indices):
    """Distributes image shards to users based on specified indices.
    
    Args:
    - dict_users: A dictionary mapping user IDs to their assigned indices.
    - idxs: An array of all available indices.
    - num_imgs: The number of images per shard.
    - shard_indices: The indices of the shards to distribute.
    """
    for shard in shard_indices:
        dict_users[i] = np.concatenate(  # Concatenate new indices to user's list
            (dict_users[i], idxs[shard * num_imgs:(shard + 1) * num_imgs]), axis=0
        )

def emnist_iid(dataset, num_users):
    """Distributes the EMNIST dataset to users in an IID manner.
    
    Args:
    - dataset: The EMNIST dataset.
    - num_users: The number of users to distribute data to.
    
    Returns:
    - A dictionary mapping user IDs to their assigned indices.
    """
    num_items_per_user = 2400  # Number of items assigned to each user
    dict_users = {i: np.array([]) for i in range(num_users)}  # Initialize user dictionary
    all_idxs = np.arange(len(dataset))  # Array of all indices
    
    for i in range(num_users):
        selected_idxs = sample_random_indices(all_idxs, num_items_per_user)  # Sample indices
        dict_users[i] = selected_idxs  # Assign sampled indices to user
        all_idxs = np.setdiff1d(all_idxs, selected_idxs)  # Remove sampled indices from available pool
    
    return dict_users  # Return user index mapping

def emnist_noniid(dataset, num_users, number_of_classes_group1_user, noniidness_end_id, mix):
    """Distributes the EMNIST dataset to users in a non-IID manner.
    
    Args:
    - dataset: The EMNIST dataset.
    - num_users: The number of users to distribute data to.
    - number_of_classes_group1_user: Number of classes per user in the first group.
    - noniidness_end_id: The endpoint for non-IID distribution.
    - mix: Indicates whether to mix the distribution.
    
    Returns:
    - A dictionary mapping user IDs to their assigned indices.
    """
    num_shards, num_imgs = 1128, 100  # Number of shards and images per shard
    dict_users = {i: np.array([]) for i in range(num_users)}  # Initialize user dictionary
    idxs = np.arange(num_shards * num_imgs)  # Create array of all indices
    labels = dataset.train_labels.numpy()  # Extract labels from dataset
    
    # Combine indices with labels and sort by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # Sorted indices based on labels
    
    if mix == 0:  # If no mixing is required
        if number_of_classes_group1_user == 0:
            # Assign shards equally to each user
            for i in range(num_users):
                shard_indices = [j for j in range(i * 24, i * 24 + 24)]  # Get shard indices
                distribute_shards(dict_users, idxs, num_imgs, shard_indices)  # Distribute shards
        else:
            already_gotten_classes = []  # Track classes already assigned
            for i in range(noniidness_end_id):
                rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
                while rand_class in already_gotten_classes:  # Ensure unique class selection
                    rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
                already_gotten_classes.append(rand_class)  # Add to tracked classes
                shard_indices = [rand_class * 24 + j for j in range(24 * number_of_classes_group1_user)]
                distribute_shards(dict_users, idxs, num_imgs, shard_indices)  # Distribute shards

            num_items = 2400
            all_idxs = np.arange(len(dataset))
            list_of_removed_classes = []

            # Remove classes already assigned to users
            for j in range(noniidness_end_id):
                list_of_removed_classes.extend(dataset.train_labels[dict_users[j]])

            set_of_removed_classes = set(list_of_removed_classes)  # Create set for quick lookup
            to_be_deleted = [idd for idd in idxs if labels[idd] in set_of_removed_classes]  # Identify indices to delete
            all_idxs = np.setdiff1d(all_idxs, to_be_deleted)  # Update available indices

            # Assign remaining users with random indices
            for i in range(noniidness_end_id, num_users):
                selected_idxs = sample_random_indices(all_idxs, num_items)
                dict_users[i] = selected_idxs  # Assign indices to user
                all_idxs = np.setdiff1d(all_idxs, selected_idxs)  # Update available indices
                
    return dict_users  # Return user index mapping

def cifar100_iid(dataset, num_users):
    """Distributes the CIFAR100 dataset to users in an IID manner.
    
    Args:
    - dataset: The CIFAR100 dataset.
    - num_users: The number of users to distribute data to.
    
    Returns:
    - A dictionary mapping user IDs to their assigned indices.
    """
    num_items_per_user = len(dataset) // num_users  # Calculate number of items per user
    dict_users = {i: np.array([]) for i in range(num_users)}  # Initialize user dictionary
    all_idxs = np.arange(len(dataset))  # Array of all indices
    
    for i in range(num_users):
        selected_idxs = sample_random_indices(all_idxs, num_items_per_user)  # Sample indices
        dict_users[i] = selected_idxs  # Assign sampled indices to user
        all_idxs = np.setdiff1d(all_idxs, selected_idxs)  # Remove sampled indices from available pool
    
    return dict_users  # Return user index mapping

def cifar100_noniid(dataset, num_users, number_of_classes_group1_user, noniidness_end_id):
    """Distributes the CIFAR100 dataset to users in a non-IID manner.
    
    Args:
    - dataset: The CIFAR100 dataset.
    - num_users: The number of users to distribute data to.
    - number_of_classes_group1_user: Number of classes per user in the first group.
    - noniidness_end_id: The endpoint for non-IID distribution.
    
    Returns:
    - A dictionary mapping user IDs to their assigned indices.
    """
    num_shards, num_imgs = 500, 100  # Number of shards and images per shard
    dict_users = {i: np.array([]) for i in range(num_users)}  # Initialize user dictionary
    idxs = np.arange(num_shards * num_imgs)  # Create array of all indices
    labels = dataset.targets  # Extract labels from dataset
    
    # Combine indices with labels and sort by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # Sorted indices based on labels
    
    already_gotten_classes = []
    for i in range(noniidness_end_id):
        rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
        while rand_class in already_gotten_classes:  # Ensure unique class selection
            rand_class = random.randint(0, noniidness_end_id * number_of_classes_group1_user - 1)
        already_gotten_classes.append(rand_class)  # Add to tracked classes
        shard_indices = [rand_class * 5 + j for j in range(5 * number_of_classes_group1_user)]
        distribute_shards(dict_users, idxs, num_imgs, shard_indices)  # Distribute shards
    
    num_items = 1000  # Number of items per user
    all_idxs = np.arange(len(dataset))
    list_of_removed_classes = []
    
    # Remove classes already assigned to users
    for j in range(noniidness_end_id):
        list_of_removed_classes.extend(dataset.targets[dict_users[j]])
    
    set_of_removed_classes = set(list_of_removed_classes)  # Create set for quick lookup
    to_be_deleted = [idd for idd in idxs if labels[idd] in set_of_removed_classes]  # Identify indices to delete
    all_idxs = np.setdiff1d(all_idxs, to_be_deleted)  # Update available indices
    
    # Assign remaining users with random indices
    for i in range(noniidness_end_id, num_users):
        selected_idxs = sample_random_indices(all_idxs,
