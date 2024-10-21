#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import (
    emnist_iid,
    emnist_noniid,
    cifar100_iid,
    cifar100_noniid,
)

def get_dataset(args):
    """
    Returns the training and test datasets, along with a user group (dictionary).
    The dictionary maps each user index to their corresponding data.

    Args:
    - args: A set of arguments specifying the dataset type, IID/non-IID setting, and other parameters.

    Returns:
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - user_groups: A dictionary mapping user IDs to their respective dataset partitions.
    """

    # Load CIFAR-100 dataset
    if args.dataset == "cifar100":
        data_dir = "../data/cifar100/"
        # Data transformations for training and testing
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Download CIFAR-100 training and test datasets
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform_test
        )

        # Sample user data: IID or non-IID
        if args.iid:
            user_groups = cifar100_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar100_noniid(
                train_dataset,
                args.num_users,
                args.number_of_classes_group1_user,
                args.noniidness_end_id,
            )

    # Load EMNIST datasets
    elif args.dataset in ["mnist", "emnist", "emnist-balanced"]:
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if args.dataset == "emnist-balanced":
            data_dir = "../data/emnist-balanced/"
            # Download EMNIST (balanced split) datasets
            train_dataset = datasets.EMNIST(
                data_dir,
                split="balanced",
                train=True,
                download=True,
                transform=apply_transform,
            )
            test_dataset = datasets.EMNIST(
                data_dir, split="balanced", train=False, download=True, transform=apply_transform
            )

            # Sample user data: IID or non-IID
            if args.iid:
                user_groups = emnist_iid(train_dataset, args.num_users)
            else:
                user_groups = emnist_noniid(
                    train_dataset,
                    args.num_users,
                    args.number_of_classes_group1_user,
                    args.noniidness_end_id,
                    args.mix,
                )

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Computes the average of the model weights across all users.
    
    Args:
    - w: A list of model weight dictionaries, where each dictionary represents the weights of a user's model.

    Returns:
    - w_avg: The averaged model weights.
    """
    w_avg = copy.deepcopy(w[0])  # Initialize with the first user's weights
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]  # Sum weights across all users
        w_avg[key] = torch.div(w_avg[key], len(w))  # Divide by the number of users to get the average
    return w_avg

def weighted_averages_n_classes(w, samples_per_class, num_classes):
    """
    Computes the weighted average of model weights, taking into account the number of samples per class.

    Args:
    - w: A list of model weight dictionaries, where each dictionary represents the weights of a user's model.
    - samples_per_class: A list of dictionaries mapping user IDs to the number of samples they have for each class.
    - num_classes: The total number of classes.

    Returns:
    - w_avg: The weighted averaged model weights.
    """
    total_samples_per_class = [0] * num_classes  # Initialize total samples for each class
    power_of_users_per_class_table = []

    # Calculate the total number of samples per class across all users
    for user_id in range(0, len(w)):
        for classs in range(0, num_classes):
            user_map = samples_per_class[user_id]
            total_samples_per_class[classs] += user_map[classs]

    # Calculate the "power" (weight) of each user for each class
    for user_id in range(0, len(w)):
        power_of_user_per_class = {}
        for classs in range(0, num_classes):
            user_map = samples_per_class[user_id]
            if total_samples_per_class[classs] != 0:
                power_of_user_per_class[classs] = user_map[classs] / total_samples_per_class[classs]
            else:
                power_of_user_per_class[classs] = 0
        power_of_users_per_class_table.append(power_of_user_per_class)

    # Calculate the total "power" for each user based on their contribution to all classes
    power_of_users = [0] * len(w)
    for user_id in range(0, len(w)):
        power_of_user_per_class = power_of_users_per_class_table[user_id]
        power_sum = 0
        for i in range(0, num_classes):
            power_sum += power_of_user_per_class[i]
        power_of_users[user_id] = power_sum

    total_power = sum(power_of_users)  # Total "power" of all users
    w_avg = copy.deepcopy(w[0])  # Initialize with the first user's weights
    for key in w_avg.keys():
        w_avg[key] = 0
        # Apply the weighted average based on user power
        for i in range(0, len(w)):
            percentage = power_of_users[i] / total_power
            w_avg[key] += w[i][key] * percentage
    return w_avg


def exp_details(args):
    """
    Prints the experimental details for the current federated learning run.

    Args:
    - args: A set of arguments specifying the model, optimizer, learning rate, epochs, and other federated parameters.
    """
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("Non-IID")
        print(f"    Local Batch size: {args.local_bs}")
        print(f"    Local Epochs: {args.local_ep}\n")
        print(f"    Averaging Type: {args.avg_type}\n")
        print(f"    Number of classes per user: {args.number_of_classes_group1_user}\n")
        print(f"    Non-IIDness end ID: {args.noniidness_end_id}\n")
        print(f"    Fraction of users: {args.frac}")
    return
