
# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        # Initialize an empty list to hold the convolutional layers.
        self.convolutional_layers = []

        # Set the initial output width to the input width. This will be updated as each layer is added.
        output_width = input_width

        # Iterate over the number of layers specified by 'self.nlayers'.
        for i in range(self.nlayers):
            # Create a convolutional layer with the specified number of input channels, 
            # number of output channels (num_channels[i]), kernel size, and stride for the current layer.
            conv = Conv1d(num_input_channels, num_channels[i], kernel_sizes[i], strides[i])
            
            # Append the created convolutional layer to the list of convolutional layers.
            self.convolutional_layers.append(conv)
            
            # Calculate the output width for the current layer based on the input width, kernel size, and stride.
            # This formula is specific to how convolution operates, considering no padding is used.
            output_width = ((output_width - kernel_sizes[i]) // strides[i]) + 1
            
            # Update the number of input channels for the next layer to be the number of output channels from the current layer.
            num_input_channels = num_channels[i]

        # Initialize the flatten layer that will be used to flatten the output of the last convolutional layer
        # before passing it to a linear layer. This is necessary because the linear layer expects a 1D input.
        self.flatten = Flatten()

        # Initialize a linear (fully connected) layer that takes the flattened output from the convolutional layers
        # (which is 'num_channels[-1] * output_width' in size) and connects it to the specified number of neurons 
        # in the linear layer ('num_linear_neurons').
        self.linear_layer = Linear(num_channels[-1] * output_width, num_linear_neurons)

        # Initialize a placeholder (self.Z) for an output or an intermediate computation that might be used later.
        # This could be used for storing the output of the network or any other value for further computations.
        self.Z = None

        # <---------------------

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Your code goes here -->
        # Iterate through each layer
        # <---------------------

        # Start with the input array 'A' as the initial output.
        out = A
        # Initialize a counter 'j' for tracking the index of activation functions.
        j = 0
        # Iterate through each convolutional layer defined in 'self.nlayers'.
        # For each layer, apply the convolution followed by the activation function.
        for i in range(self.nlayers):
            # Apply the convolution operation for the i-th layer.
            out = self.convolutional_layers[i].forward(out)
            # Apply the activation function corresponding to the i-th layer.
            out = self.activations[j].forward(out)
            # Increment the activation function index.
            j += 1
        
        # Flatten the output from the last convolutional layer to make it suitable for the linear layer.
        out = self.flatten.forward(out)

        # Pass the flattened output through the linear layer.
        out = self.linear_layer.forward(out)

        # Store the final output in 'self.Z' for potential future use.
        self.Z = out
        
        # Return the final output.
        return self.Z
            

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        # Calculate the number of examples ('m') in the batch and ignore the other dimension (number of linear neurons).
        m, _ = labels.shape

        # Compute the loss by applying the forward method of the criterion (loss function) to the network's output and the labels,
        # and then summing up the loss over all examples in the batch.
        self.loss = self.criterion.forward(self.Z, labels).sum()

        # Compute the gradient of the loss with respect to the output of the network by applying the backward method of the criterion.
        grad = self.criterion.backward()

        # First, propagate gradients through the linear layer in reverse order.
        grad = self.linear_layer.backward(grad)

        # Then, propagate gradients through the flatten layer to transform them back to the format expected by the convolutional layers.
        grad = self.flatten.backward(grad)

        # Initialize the index 'i' to point to the last activation function in the list.
        i = len(self.convolutional_layers) - 1

        # Iterate through each convolutional layer in reverse order to propagate the gradients.
        # Note: The loop iterates over convolutional layers and their corresponding activation functions in reverse.
        for layer in self.convolutional_layers[::-1]:
            # Apply the backward method of the corresponding activation function to update the gradients.
            grad = grad * self.activations[i].backward()
            
            # Propagate gradients through the current convolutional layer.
            grad = layer.backward(grad)

            # Decrement the index 'i' to move to the previous activation function for the next iteration.
            i -= 1

        # Return the computed gradient with respect to the input, after completing the backward pass through all layers.
        return grad
        
    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
