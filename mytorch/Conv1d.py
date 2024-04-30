# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)


        
    def forward(self, A):
        """
        Argument:
    
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Store the input tensor for use in the backward pass
        self.A = A

        # Initialize the output tensor Z with zeros. The output size is computed based on the input size, 
        # the size of the filters (self.W.shape[2]), and the kernel size
        Z = np.zeros((A.shape[0], self.W.shape[0], A.shape[2] - self.kernel_size + 1))

        # Perform the convolution operation by sliding the filters across the input tensor
        for i in range(A.shape[2] - self.kernel_size + 1):
            Z[:, :, i] = np.tensordot(A[:, :, i:i+self.W.shape[2]], self.W, axes=[(1, 2), (1, 2)])
        
        # Duplicate the bias term for each output feature map and add it to the result
        b = np.vstack([self.b]*Z.shape[2])
        b_trans = b.T
        Z = Z + b_trans

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        input_size = self.A.shape[2]

        # Determine the padding needed based on the kernel size
        pad_zeros = (self.kernel_size - 1) * 2
        # Flip the filters for the convolution operation
        flipped_W = np.flip(self.W, axis=2)

        # Initialize the gradient with respect to the input as zeros
        dLdA = np.zeros((dLdZ.shape[0], self.W.shape[1], input_size))
        # Pad the gradients with respect to the output tensor to maintain the size after convolution
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (pad_zeros//2, pad_zeros//2)), mode='constant', constant_values=0)
        
        # Perform the convolution with the flipped filters to compute dLdA
        for i in range(dLdA.shape[2]):
            dLdA[:, :, i] = np.tensordot(padded_dLdZ[:, :, i:i+flipped_W.shape[2]], flipped_W, axes=[(1, 2), (0, 2)])

        # Initialize the gradient with respect to the filters as zeros
        self.dLdW = np.zeros(self.W.shape)
        # Compute the gradient with respect to the filters
        for i in range(self.dLdW.shape[2]):
            self.dLdW[:, :, i] = np.tensordot(self.A[:, :, i:i+dLdZ.shape[2]], dLdZ, axes=[(0, 2), (0, 2)]).T

        # Compute the gradient with respect to the bias by summing over the batch and output size dimensions
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        
        return dLdA




class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Apply a 1D convolution to the input tensor. The convolution operation is encapsulated in the 'conv1d_stride1' object.
        Z = self.conv1d_stride1.forward(A)
        
        # Apply downsampling to the output of the convolution. The downsampling operation is encapsulated in the 'downsample1d' object.
        # Downsampling may reduce the temporal resolution of the output, depending on the stride configuration.
        Z_down = self.downsample1d.forward(Z)

        return Z_down

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Compute the gradient through the downsampling layer. This operation will 'upsample' the gradient to match
        # the size before downsampling, which is necessary for correct gradient computation through the convolution layer.
        dLdZ_temp = self.downsample1d.backward(dLdZ)

        # Compute the gradient through the 1D convolution layer using the gradient from the downsampling layer.
        # This step will compute how much each element of the input tensor contributed to the loss.
        dLdA = self.conv1d_stride1.backward(dLdZ_temp)

        return dLdA
