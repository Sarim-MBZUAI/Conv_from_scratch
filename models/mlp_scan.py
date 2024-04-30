# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')



class CNN_SimpleScanningMLP():
    def __init__(self):
        # Initialize the first convolutional layer with 24 input channels, 8 output channels,
        # a kernel size of 8, and a stride of 4. This configuration defines how the convolution
        # will scan the input data.
        self.conv1 = Conv1d(24, 8, kernel_size=8, stride=4)  # TODO

        # Initialize the second convolutional layer with 8 input channels (the output of conv1),
        # 16 output channels, a kernel size of 1, and a stride of 1. This layer will apply a
        # 1x1 convolution, which can be used for channel-wise transformations.
        self.conv2 = Conv1d(8, 16, kernel_size=1, stride=1)  # TODO

        # Initialize the third convolutional layer with 16 input channels (the output of conv2),
        # 4 output channels, a kernel size of 1, and a stride of 1. Similar to conv2, this layer
        # also applies a 1x1 convolution for channel-wise transformations.
        self.conv3 = Conv1d(16, 4, kernel_size=1, stride=1)  # TODO

        # Organize the layers of the network in sequence, including ReLU activation functions
        # after conv1 and conv2, and a Flatten layer at the end to prepare the output for
        # further processing (e.g., a fully connected layer).
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):
        # Unpack the provided weights tuple into individual weight arrays for each convolutional layer.
        w1, w2, w3 = weights

        # Initialize weights for conv1 by reshaping the provided weight array to match
        # the expected dimensions (8, 24, 8), and then transpose the dimensions to fit
        # the expected layout for the convolution operation.
        self.conv1.conv1d_stride1.W = w1.reshape(8, 24, 8).transpose([2, 1, 0]) # TODO

        # Initialize weights for conv2 by reshaping the provided weight array to match
        # the expected dimensions (1, 8, 16), and then transpose the dimensions. This step
        # prepares the 1x1 convolution weights for channel-wise transformations.
        self.conv2.conv1d_stride1.W = w2.reshape(1, 8, 16).transpose([2, 1, 0]) # TODO

        # Initialize weights for conv3 by reshaping the provided weight array to match
        # the expected dimensions (1, 16, 4), and then transpose the dimensions. This final
        # step sets up the 1x1 convolution weights for the last convolutional layer.
        self.conv3.conv1d_stride1.W = w3.reshape(1, 16, 4).transpose([2, 1, 0]) # TODO



    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # self.conv1 = None
        # self.conv2 = None
        # self.conv3 = None
        # self.layers = [] # TODO: Add the layers in the correct order

        # First convolutional layer with 24 input channels, 2 output channels, kernel size 2, and stride 2.
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2) #  TODO

        # Second convolutional layer with 2 input channels, 8 output channels, kernel size 2, and stride 2.
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2) # TODO

        # Third convolutional layer with 8 input channels, 4 output channels, kernel size 2, and stride 1.
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1) # TODO

        # Layers list includes convolutional layers, ReLU activations, and a Flatten layer at the end.
        self.layers = [
            self.conv1, ReLU(),  # Conv1 + Activation
            self.conv2, ReLU(),  # Conv2 + Activation
            self.conv3, Flatten()  # Conv3 + Flatten
        ]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # Decompose weights into individual matrices for each convolutional layer.
        w1, w2, w3 = weights

        # Set conv1 weights: Transpose w1, select and reshape to (2, 8, 24), slice, and transpose to fit conv1's dimensions.
        self.conv1.conv1d_stride1.W = w1.T[:2, :].reshape(2, 8, 24)[:, :2, :].transpose([0, 2, 1])

        # Set conv2 weights: Transpose w2, select and reshape to (8, 4, 2), slice, and transpose to fit conv2's dimensions.
        self.conv2.conv1d_stride1.W = w2.T[:8, :].reshape(8, 4, 2)[:, :2, :].transpose([0, 2, 1])

        # Set conv3 weights: Transpose w3, reshape to (4, 2, 8), and transpose to fit conv3's dimensions.
        self.conv3.conv1d_stride1.W = w3.T.reshape(4, 2, 8).transpose([0, 2, 1])

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
