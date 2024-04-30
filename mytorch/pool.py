import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Store the input tensor for use in the backward pass.
        self.A = A
        # Unpack the dimensions of the input tensor.
        batch_size, in_channels, input_height, input_width = A.shape
        # Calculate the dimensions of the output tensor.
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        # Initialize the output tensor and a tensor to store the indices of the max values within each kernel window.
        self.Z = np.zeros((batch_size, in_channels, output_height, output_width))
        self.max_indices = np.zeros((batch_size, in_channels, output_height, output_width, 2), dtype=int)

        # Iterate over the output tensor dimensions and apply the max pooling operation.
        for i in range(output_height):
            for j in range(output_width):
                # Extract the current kernel window across all batches and channels.
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                # Compute the max value within the window.
                self.Z[:, :, i, j] = np.max(window, axis=(2, 3))
                # Find the indices of the max values and store them.
                max_pos = np.argmax(window.reshape(batch_size, in_channels, -1), axis=2)
                self.max_indices[:, :, i, j, 0] = max_pos // self.kernel
                self.max_indices[:, :, i, j, 1] = max_pos % self.kernel

        return self.Z
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Initialize the gradient tensor with respect to the input as zeros.
        dLdA = np.zeros_like(self.A)

        # Unpack the dimensions of the gradient tensor.
        batch_size, in_channels, output_height, output_width = dLdZ.shape

        # Iterate over the output tensor dimensions to map the gradients back to the input tensor.
        for i in range(output_height):
            for j in range(output_width):
                # Retrieve the indices of the max values within each kernel window.
                max_i = self.max_indices[:, :, i, j, 0] + i
                max_j = self.max_indices[:, :, i, j, 1] + j
                # Update the gradient of the input tensor at the positions of the max values.
                dLdA[np.arange(batch_size)[:, None], np.arange(in_channels), max_i, max_j] += dLdZ[:, :, i, j]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Store the input tensor for use in the backward pass.
        self.A = A
        # Unpack the dimensions of the input tensor.
        batch_size, in_channels, input_height, input_width = A.shape
        # Calculate the dimensions of the output tensor.
        output_height = input_height - self.kernel + 1
        output_width = input_width - self.kernel + 1

        # Initialize the output tensor for the mean pooled values.
        self.Z = np.zeros((batch_size, in_channels, output_height, output_width))
        # Iterate over the output tensor dimensions and apply the mean pooling operation.
        for i in range(output_height):
            for j in range(output_width):
                # Compute the mean value within each kernel window and store it in the output tensor.
                self.Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return self.Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Initialize the gradient tensor with respect to the input as zeros.
        dLdA = np.zeros_like(self.A)
        # Unpack the dimensions of the gradient tensor.
        batch_size, in_channels, output_height, output_width = dLdZ.shape

        # Iterate over the output tensor dimensions to distribute the gradients back to the input tensor.
        for i in range(output_height):
            for j in range(output_width):
                # Each element in the kernel window contributes equally to the mean,
                # so the gradient is distributed evenly across the kernel window.
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel**2)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # First, apply max pooling with a stride of 1 to the input tensor.
        Z = self.maxpool2d_stride1.forward(A)
        
        # If the stride is set to a value greater than 1, additionally apply downsampling to the result of max pooling.
        if self.stride > 1:
            Z = self.downsample2d.forward(Z)  # Applying downsampling if stride is greater than 1
        
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # If downsampling was applied in the forward pass, apply its backward method first to propagate the gradients.
        if self.stride > 1:
            dLdZ = self.downsample2d.backward(dLdZ)
        
        # Finally, propagate gradients through the max pooling layer with a stride of 1.
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Initialize an instance of MeanPool2d_stride1 to perform mean pooling with a stride of 1.
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        # Initialize an instance of Downsample2d to perform downsampling if stride > 1.
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Apply mean pooling with a stride of 1 to the input tensor.
        Z = self.meanpool2d_stride1.forward(A)

        # If the stride is greater than 1, additionally apply downsampling to the result of mean pooling.
        if self.stride > 1:
            Z = self.downsample2d.forward(Z)  # Applying downsampling if stride is greater than 1

        return Z



    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # If downsampling was applied in the forward pass, apply its backward method first to propagate the gradients.
        if self.stride > 1:
            dLdZ = self.downsample2d.backward(dLdZ)

        # Finally, propagate gradients through the mean pooling layer with a stride of 1.
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
