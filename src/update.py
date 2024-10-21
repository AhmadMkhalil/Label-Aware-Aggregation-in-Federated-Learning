#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class that wraps around the PyTorch Dataset class.
    
    This class is used to create a subset of a dataset based on given indices.
    """

    def __init__(self, dataset, idxs):
        """
        Initializes the DatasetSplit instance.
        
        Args:
        - dataset: The original dataset to split.
        - idxs: A list of indices corresponding to the data points in the dataset.
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]  # Convert indices to integers

    def __len__(self):
        """Returns the number of items in the dataset subset."""
        return len(self.idxs)

    def __getitem__(self, item):
        """Fetches the item at the specified index from the dataset.
        
        Args:
        - item: Index of the item to fetch.
        
        Returns:
        - A tuple (image, label) as tensors.
        """
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)  # Return tensors


class LocalUpdate(object):
    """Handles local model updates and training for a given user.
    
    This class manages the data loaders and training procedure for local model updates.
    """

    def __init__(self, args, dataset, idxs, logger):
        """
        Initializes the LocalUpdate instance.
        
        Args:
        - args: A set of arguments specifying configuration parameters.
        - dataset: The complete dataset to sample from.
        - idxs: Indices of the data assigned to the user.
        - logger: A logger for tracking training progress and metrics.
        """
        self.args = args
        self.logger = logger
        # Create data loaders for training, validation, and testing
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'  # Set device to GPU or CPU
        self.criterion = nn.CrossEntropyLoss()  # Define loss function for training

    def train_val_test(self, dataset, idxs):
        """Splits the dataset into training, validation, and test sets, and returns their data loaders.
        
        Args:
        - dataset: The original dataset.
        - idxs: Indices of the data for the current user.
        
        Returns:
        - trainloader: DataLoader for the training set.
        - validloader: DataLoader for the validation set (currently None).
        - testloader: DataLoader for the test set.
        """
        # Split indices into training and test sets (80% train, 20% test)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]

        # Create data loaders for training and testing
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = None  # Validation loader not implemented
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model):
        """Trains the model using the local training data and updates the weights.
        
        Args:
        - model: The neural network model to be trained.
        
        Returns:
        - A tuple containing the updated model's state dictionary and average epoch loss.
        """
        model.train()  # Set model to training mode
        epoch_loss = []  # List to store loss values for each epoch

        # Choose optimizer based on user argument
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # Perform local training for a specified number of epochs
        for _ in range(self.args.local_ep):
            batch_loss = []  # List to store loss values for the current batch
            for _, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)  # Move data to device

                optimizer.zero_grad()  # Zero the gradients before backward pass
                log_probs = model(images)  # Forward pass
                loss = self.criterion(log_probs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                batch_loss.append(loss.item())  # Record batch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))  # Average loss for the epoch

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)  # Return updated weights and average loss

    def inference(self, model):
        """Evaluates the model on the test dataset and calculates accuracy and loss.
        
        Args:
        - model: The trained model to evaluate.
        
        Returns:
        - A tuple containing accuracy and total loss on the test dataset.
        """
        model.eval()  # Set model to evaluation mode
        loss, total, correct = 0.0, 0.0, 0.0  # Initialize metrics

        # Evaluate the model on the test dataset
        for _, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)  # Move data to device

            outputs = model(images)  # Forward pass
            batch_loss = self.criterion(outputs, labels)  # Compute loss
            loss += batch_loss.item()  # Accumulate loss

            # Get predictions and calculate accuracy
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)  # Flatten predictions
            correct += torch.sum(torch.eq(pred_labels, labels)).item()  # Count correct predictions
            total += len(labels)  # Total number of labels processed

        accuracy = correct / total  # Calculate accuracy
        return accuracy, loss  # Return accuracy and loss


def test_inference(args, model, test_dataset):
    """Evaluates the model on a provided test dataset and calculates accuracy and loss.
    
    Args:
    - args: A set of arguments specifying configuration parameters.
    - model: The trained model to evaluate.
    - test_dataset: The dataset to test the model on.
    
    Returns:
    - A tuple containing accuracy and total loss on the test dataset.
    """
    model.eval()  # Set model to evaluation mode
    loss, total, correct = 0.0, 0.0, 0.0  # Initialize metrics

    device = 'cuda' if args.gpu else 'cpu'  # Set device for evaluation
    criterion = nn.NLLLoss().to(device)  # Define loss function for testing
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # Create test DataLoader

    # Evaluate the model on the test dataset
    for _, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)  # Move data to device

        outputs = model(images)  # Forward pass
        batch_loss = criterion(outputs, labels)  # Compute loss
        loss += batch_loss.item()  # Accumulate loss

        # Get predictions and calculate accuracy
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)  # Flatten predictions
        correct += torch.sum(torch.eq(pred_labels, labels)).item()  # Count correct predictions
        total += len(labels)  # Total number of labels processed

    accuracy = correct / total  # Calculate accuracy
    return accuracy, loss  # Return accuracy and loss
