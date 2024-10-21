#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    """Parses command-line arguments for the federated learning setup."""
    
    parser = argparse.ArgumentParser()

    # Federated learning arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of rounds of training.")
    parser.add_argument('--num_users', type=int, default=10,
                        help="Total number of users participating in federated learning.")
    parser.add_argument('--frac', type=float, default=0.3,
                        help='Fraction of users selected in each round (k_r).')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="Number of local epochs to train on each user.")
    parser.add_argument('--local_bs', type=int, default=256,
                        help="Local batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimization (default: 0.001).')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimization (default: 0.9).')

    # Model arguments
    parser.add_argument('--model', type=str, default='cnn', help='Name of the model to use (e.g., cnn).')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='Number of each kind of kernel used in convolution.')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='Comma-separated sizes of kernels to use for convolution.')
    parser.add_argument('--num_channels', type=int, default=3,
                        help="Number of channels in input images (e.g., 3 for RGB).")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="Normalization type: batch_norm, layer_norm, or None.")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="Number of filters for convolutional networks (32 for mini-imagenet, 64 for omiglot).")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether to use max pooling rather than strided convolutions.")

    # Other arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help="Name of the dataset (e.g., emnist-balanced, cifar100).")
    parser.add_argument('--num_classes', type=int, default=100,
                        help="Number of classes in the dataset (e.g., 47 or 100).")
    parser.add_argument('--gpu', default=1,
                        help="Specify GPU ID to use CUDA; set to CPU if not using CUDA.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="Type of optimizer to use (e.g., sgd, adam).")
    parser.add_argument('--iid', type=int, default=0,
                        help='Set to 0 for non-IID data distribution, default is IID (1).')
    parser.add_argument('--unequal', type=int, default=0,
                        help='Set to 1 for unequal data splits in non-IID setting (default: 0 for equal splits).')
    parser.add_argument('--number_of_classes_group1_user', type=int, default=1,
                        help='Unique count of classes per user in the first group.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='Number of rounds for early stopping mechanism.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level for logging (0 for silent, 1 for verbose).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility.')
    parser.add_argument('--avg_type', type=str, default='avg',
                        help='Average type for federated averaging: "avg" or "avg_n_classes".')
    parser.add_argument('--noniidness_end_id', type=int, default=3,
                        help='End ID for controlling non-IID data distribution.')                    

    args = parser.parse_args()  # Parse the arguments
    return args  # Return the parsed arguments
