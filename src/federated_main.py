#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from datetime import datetime

from options import args_parser  # Argument parser for command-line options
from update import LocalUpdate, test_inference  # Local training and testing functions
from models import (
    CNNEmnist,  # CNN model for the EMNIST dataset
    CNNCifar100,  # CNN model for the CIFAR-100 dataset
)
from utils import (
    get_dataset,  # Function to load datasets
    average_weights,  # Function to average model weights
    exp_details,  # Function to print experiment details
    weighted_averages_n_classes,  # Function for label-aware averaging
)

if __name__ == "__main__":
    start_time = time.time()  # Start timer for performance measurement

    # Define paths
    path_project = os.path.abspath("..")  # Get the absolute path of the project
    logger = None  # Placeholder for TensorBoard logger (currently unused)

    # Parse command-line arguments
    args = args_parser()
    exp_details(args)  # Print experiment details

    # Set the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")  # Get current timestamp for logging

    # Log user group details
    with open(
        f"user_groups_{args.avg_type}_clsg1{args.number_of_classes_group1_user}_endid{args.noniidness_end_id}_frac{args.frac}_{current_time}.txt",
        "w",
    ) as f:
        for userId in range(len(user_groups)):
            print(f"UserId: {userId}", file=f)  # Log user ID
            samples_ids = [int(s) for s in user_groups[userId]]
            samples_ids.sort()  # Sort sample IDs for consistency

            # Get and sort sample classes for the current user
            samples_classes = [
                int(train_dataset.train_labels[i]) if args.dataset not in ("cifar", "cifar100")
                else int(train_dataset.targets[i])
                for i in samples_ids
            ]
            samples_classes.sort()

            # Log sample size and classes
            print(f"Samples Size: {len(samples_classes)}", file=f)
            print(f"Samples Classes: {set(samples_classes)}", file=f)

    # BUILD MODEL
    if args.model == "cnn":
        # Choose model based on dataset
        if args.dataset == "emnist-balanced":
            global_model = CNNEmnist(args=args)
        elif args.dataset == "cifar100":
            global_model = CNNCifar100(args=args)
    else:
        exit("Error: Unrecognized model")

    # Move model to the appropriate device and set to training mode
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Copy initial weights from the global model
    global_weights = global_model.state_dict()

    # Initialize lists to track losses and accuracies
    train_loss, train_accuracy = [], []
    test_loss_list, test_accuracy_list = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1  # Frequency of logging results
    val_loss_pre, counter = 0, 0  # Variables for early stopping logic

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []  # Initialize lists for local weights and losses
        data, samples_per_class = [], []  # Lists to store user data statistics
        classes = []  # List to keep track of classes

        print(f"\n| Global Training Round: {epoch + 1} |\n")

        global_model.train()  # Set model to training mode
        m = max(int(args.frac * args.num_users), 1)  # Determine number of users to sample
        idxs_users = np.random.choice(
            range(args.num_users), m, replace=False  # Randomly select users for this round
        )
        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],  # User's data indices
                logger=logger,
            )
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model))  # Update weights locally
            local_weights.append(copy.deepcopy(w))  # Store local weights
            local_losses.append(copy.deepcopy(loss))  # Store local loss
            user_data = len(user_groups[idx])  # Number of samples for this user
            data.append(user_data)  # Collect user data statistics

            # Count samples per class for current user
            user_samples_per_class = {i: 0 for i in range(args.num_classes)}  # Initialize class count
            if args.dataset not in ("cifar", "cifar100"):
                for i in user_groups[idx]:
                    class_label = int(train_dataset.train_labels[i])
                    classes.append(class_label)
                    user_samples_per_class[class_label] += 1  # Increment count for the respective class
            else:
                for i in user_groups[idx]:
                    class_label = int(train_dataset.targets[i])
                    classes.append(class_label)
                    user_samples_per_class[class_label] += 1  # Increment count for the respective class

            samples_per_class.append(user_samples_per_class)  # Store counts for this user

        # Update global weights using either simple average or weighted average based on classes
        if args.avg_type == "avg":
            global_weights = average_weights(local_weights)  # FedAVG
        else:
            # Use label-aware averaging
            global_weights = weighted_averages_n_classes(
                local_weights, samples_per_class, args.num_classes
            )

        # Load the averaged weights into the global model
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)  # Average loss from local updates
        train_loss.append(round(loss_avg, 3))  # Store average training loss

        # Calculate average training accuracy across all users at every epoch
        list_acc, list_loss = [], []  # Lists for tracking accuracy and loss
        for c in range(args.num_users):
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[c],  # Use each user's data for evaluation
                logger=logger,
            )
            acc, loss = local_model.inference(model=global_model)  # Inference for accuracy
            list_acc.append(acc)  # Store accuracy
            list_loss.append(loss)  # Store loss
        train_accuracy.append(
            round((sum(list_acc) / len(list_acc)), 3)  # Average training accuracy
        )

        # Test inference after completing training for this round
        test_acc, test_loss = test_inference(args, global_model, test_dataset)  # Evaluate on the test set
        test_accuracy_list.append(test_acc)  # Store test accuracy
        test_loss_list.append(test_loss / 100)  # Store test loss (scaled)

        # Print global training loss and accuracy after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(
                f"\nAvg Training Stats after {epoch + 1} global rounds:"
            )
            print(
                f"Train Accuracy: {100 * train_accuracy[-1]:.2f}%"
            )
            print(
                f"Training Loss: {np.mean(np.array(train_loss)):.3f}"
            )
            print(
                f"Test Accuracy: {100 * test_acc:.2f}%"
            )
            print(
                f"Test Loss: {round((test_loss / 100), 3)}"
            )

    #######################   PLOTTING & args & results saving    ###################################

    plt.switch_backend("Agg")  # Use non-interactive backend for plotting

    # Determine if data is IID or non-IID for file naming
    iidness = "iid" if args.iid == 1 else "noniid"
    my_path = os.getcwd()  # Get current working directory
    full_path = os.path.join(
        my_path,
        f"../save/{args.dataset}/{iidness}/{args.avg_type}/{args.epochs}/{args.number_of_classes_group1_user}/{args.noniidness_end_id}/{args.frac}/{current_time}",
    )
    os.makedirs(full_path, exist_ok=True)  # Create directory for saving results

    # Plot Training Loss vs Communication rounds
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(train_loss)), train_loss, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(f"{full_path}/training_loss.pdf")  # Save loss plot

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("
