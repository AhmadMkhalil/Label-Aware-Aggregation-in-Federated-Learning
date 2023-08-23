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
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == "cifar100":
        data_dir = "../data/cifar100/"
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

        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform_train
        )

        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform_test
        )

        # sample training data amongst users
        if args.iid:
            user_groups = cifar100_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar100_noniid(
                train_dataset,
                args.num_users,
                args.number_of_classes_group1_user,
                args.noniidness_end_id,
            )

    elif args.dataset in ["mnist", "emnist", "emnist-balanced"]:
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    
        if args.dataset == "emnist-balanced":
            data_dir = "../data/emnist-balanced/"
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
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from EMNIST
                user_groups = emnist_iid(train_dataset, args.num_users)
            else:
                # Choose equal splits for every user
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
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def weighted_averages_n_classes(w, samples_per_class, num_classes):
    """
    Returns the average of the weights.
    """
    total_samples_per_class = [0] * num_classes
    power_of_users_per_class_table = []
    for user_id in range(0, len(w)):
        for classs in range(0, num_classes):
            user_map = samples_per_class[user_id]
            total_samples_per_class[classs] += user_map[classs]
    for user_id in range(0, len(w)):
        power_of_user_per_class = {}
        for classs in range(0, num_classes):
            user_map = samples_per_class[user_id]
            if total_samples_per_class[classs] != 0:
                power_of_user_per_class[classs] = user_map[classs] / total_samples_per_class[classs]
            else:
                power_of_user_per_class[classs] = 0
        power_of_users_per_class_table.append(power_of_user_per_class)
    power_of_users = [0] * len(w)
    for user_id in range(0, len(w)):
        power_of_user_per_class = power_of_users_per_class_table[user_id]
        power_sum = 0
        for i in range(0, num_classes):
            power_sum += power_of_user_per_class[i]
        power_of_users[user_id] = power_sum
    total_power = sum(power_of_users)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(0, len(w)):
            percentage = power_of_users[i] / total_power
            w_avg[key] += w[i][key] * percentage
    return w_avg


def exp_details(args):
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
        print(f"    number of classes group1 user: {args.number_of_classes_group1_user}\n")
        print(f"    Non-iidness-end-id: {args.noniidness_end_id}\n")
        print(f"    Fraction of users: {args.frac}")
    return
