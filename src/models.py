#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class CNNEmnist(nn.Module):
    """Convolutional Neural Network model for the EMNIST dataset."""
    
    def __init__(self, args):
        super(CNNEmnist, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)  # First convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Second convolutional layer
        self.conv2_drop = nn.Dropout2d()  # Dropout layer to reduce overfitting
        self.fc1 = nn.Linear(320, 50)  # First fully connected layer
        self.fc2 = nn.Linear(50, args.num_classes)  # Output layer (number of classes)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Apply conv1, relu activation, and max pooling
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # Apply conv2, dropout, relu, and max pooling
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))  # Apply the first fully connected layer with relu activation
        x = F.dropout(x, training=self.training)  # Apply dropout
        x = self.fc2(x)  # Apply the output layer
        return F.log_softmax(x, dim=1)  # Apply log softmax to get log probabilities


class CNNCifar100(nn.Module):
    """Convolutional Neural Network model for the CIFAR-100 dataset."""
    
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        
        # Define a reusable convolutional block
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolutional layer
                nn.BatchNorm2d(out_channels),  # Batch normalization
                nn.ReLU(inplace=True)  # ReLU activation
            ]
            if pool: 
                layers.append(nn.MaxPool2d(2))  # Add max pooling if specified
            return nn.Sequential(*layers)  # Return the sequential block

        # Define the convolutional layers and residual connections
        self.conv1 = conv_block(args.num_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)  # Second convolution with pooling
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))  # Residual block 1
        
        self.conv3 = conv_block(128, 256, pool=True)  # Third convolution with pooling
        self.conv4 = conv_block(256, 512, pool=True)  # Fourth convolution with pooling
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))  # Residual block 2
        
        self.conv5 = conv_block(512, 1028, pool=True)  # Fifth convolution with pooling
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))  # Residual block 3
        
        # Define the classifier that produces the output
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),  # Final pooling layer to reduce size
            nn.Flatten(),  # Flatten the tensor to 1D for the fully connected layer
            nn.Linear(1028, args.num_classes)  # Fully connected layer to output the number of classes
        )

    def forward(self, xb):
        """Forward pass through the network."""
        out = self.conv1(xb)  # Pass input through conv1
        out = self.conv2(out)  # Pass through conv2
        out = self.res1(out) + out  # Add residual connection for conv2
        out = self.conv3(out)  # Pass through conv3
        out = self.conv4(out)  # Pass through conv4
        out = self.res2(out) + out  # Add residual connection for conv4
        out = self.conv5(out)  # Pass through conv5
        out = self.res3(out) + out  # Add residual connection for conv5
        out = self.classifier(out)  # Pass through the classifier
        return F.log_softmax(out, dim=1)  # Apply log softmax to get log probabilities
