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

from options import args_parser
from update import LocalUpdate, test_inference
from models import (
    CNNEmnist,
    CNNCifar100,
)
from utils import (
    get_dataset,
    average_weights,
    exp_details,
    weighted_averages_n_classes,
)

if __name__ == "__main__":
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath("..")
    logger = None  # tensorboardX.SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset, test_dataset, user_groups = get_dataset(args)
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    with open(
        f"user_groups_{args.avg_type}_clsg1{args.number_of_classes_group1_user}_endid{args.noniidness_end_id}_frac{args.frac}_{current_time}.txt",
        "w",
    ) as f:
        for userId in range(0, len(user_groups)):
            print(f"UserId: {userId}", file=f)
            samples_ids = [int(s) for s in user_groups[userId]]
            samples_ids.sort()

            samples_classes = [
                int(train_dataset.train_labels[i]) if args.dataset not in ("cifar", "cifar100")
                else int(train_dataset.targets[i])
                for i in samples_ids
            ]
            samples_classes.sort()

            print(f"Samples Size: {len(samples_classes)}", file=f)
            print(f"Samples Classes: {set(samples_classes)}", file=f)

    # BUILD MODEL
    if args.model == "cnn":
        # Convolutional neural network
        if args.dataset == "emnist-balanced":
            global_model = CNNEmnist(args=args)
        elif args.dataset == "cifar100":
            global_model = CNNCifar100(args=args)
    else:
        exit("Error: Unrecognized model")

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss_list, test_accuracy_list = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        data, samples_per_class = [], []
        classes = []
        print(f"\n| Global Training Round: {epoch + 1} |\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(
            range(args.num_users), m, replace=False
        )
        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            user_data = len(user_groups[idx])
            data.append(user_data)

            ids = user_groups[idx]
            user_samples_per_class = {i: 0 for i in range(args.num_classes)}
            if args.dataset not in ("cifar", "cifar100"):
                for i in ids:
                    class_label = int(train_dataset.train_labels[i])
                    classes.append(class_label)
                    user_samples_per_class[class_label] += 1
            else:
                for i in ids:
                    class_label = int(train_dataset.targets[i])
                    classes.append(class_label)
                    user_samples_per_class[class_label] += 1

            samples_per_class.append(user_samples_per_class)

        # Update global weights
           # FedAVG
        if args.avg_type == "avg":
            global_weights = average_weights(local_weights)
        else:
            # label-aware
            global_weights = weighted_averages_n_classes(
                local_weights, samples_per_class, args.num_classes
            )

        previous_model = global_model
        previous_test_accuracy = (
            test_accuracy_list[-1]
            if len(test_accuracy_list) > 0
            else 0
        )
        previous_test_loss = (
            test_loss_list[-1] if len(test_loss_list) > 0 else 0
        )

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(round(loss_avg, 3))

        # Calculate average training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        for c in range(args.num_users):
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(
            round((sum(list_acc) / len(list_acc)), 3)
        )

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss / 100)

        # Print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(
                f"\nAvg Training Stats after {epoch + 1} global rounds:"
            )
            print(
                f"Train Accuracy: {100 * train_accuracy[-1]:.2f}%"
            )
            print(
                f"Training Loss: {np.mean(np.array(train_loss))}"
            )
            print(
                f"Test Accuracy: {100 * test_acc:.2f}%"
            )
            print(
                f"Test Loss: {round((test_loss / 100), 3)}"
            )

    #######################   PLOTTING & args & results saving    ###################################

    plt.switch_backend("Agg")

    if args.iid == 1:
        iidness = "iid"
    elif args.iid == 0:
        iidness = "noniid"
    my_path = os.getcwd()
    full_path = os.path.join(
        my_path,
        f"../save/{args.dataset}/{iidness}/{args.avg_type}/{args.epochs}/{args.number_of_classes_group1_user}/{args.noniidness_end_id}/{args.frac}/{current_time}",
    )
    os.makedirs(full_path, exist_ok=True)

    # Plot Loss curve
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(train_loss)), train_loss, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(f"{full_path}/training_loss.pdf")

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(f"{full_path}/training_accuracy.pdf")

    plt.figure()
    plt.title("Testing Loss vs Communication rounds")
    plt.plot(range(len(test_loss_list)), test_loss_list, color="r")
    plt.ylabel("Testing loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(f"{full_path}/testing_loss.pdf")

    plt.figure()
    plt.title("Testing Accuracy vs Communication rounds")
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, color="k")
    plt.ylabel("Testing Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(f"{full_path}/testing_accuracy.pdf")

    # YAML file with all data and results
    data = {
        "epochs": args.epochs,
        "num_users": args.num_users,
        "frac": args.frac,
        "local_ep": args.local_ep,
        "local_bs": args.local_bs,
        "lr": args.lr,
        "momentum": args.momentum,
        "model": args.model,
        "kernel_num": args.kernel_num,
        "kernel_sizes": args.kernel_sizes,
        "num_channels": args.num_channels,
        "norm": args.norm,
        "num_filters": args.num_filters,
        "max_pool": args.max_pool,
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "optimizer": args.optimizer,
        "iid": args.iid,
        "unequal": args.unequal,
        "verbose": args.verbose,
        "seed": args.seed,
        "avg_type": args.avg_type,
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
        "avg_train_accuracy": round(train_accuracy[-1], 3),
        "avg_train_loss": round((train_loss[-1] / 100), 3),
        "test_accuracy_list": test_accuracy_list,
        "test_loss_list": test_loss_list,
        "number_of_classes_of_half_of_user": args.number_of_classes_group1_user,
    }

    with open(f"{full_path}/data.yml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
