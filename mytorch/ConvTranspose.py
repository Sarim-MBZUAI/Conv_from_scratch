import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor  # Store the upsampling factor.

        # Initialize an Upsample1d instance to upsample the input tensor before applying the convolution.
        self.upsample1d = Upsample1d(upsampling_factor)
        
        # Initialize a Conv1d instance with stride 1 to apply convolution to the upsampled tensor.
        self.conv1d_stride1 = Conv1d(in_channels, out_channels, kernel_size, stride=1,
                                     weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # First, upsample the input tensor using the specified upsampling factor.
        A_upsampled = self.upsample1d.forward(A)

        # Then, apply a 1D convolution with stride 1 to the upsampled tensor.
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # First, compute the gradient through the convolution layer using its backward method.
        delta_out = self.conv1d_stride1.backward(dLdZ)

        # Then, propagate the gradient through the upsampling layer to compute the gradient with respect to the original input tensor.
        dLdA = self.upsample1d.backward(delta_out)

        return dLdA


    

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor
        # Initialize the 2D upsampling layer with the given upsampling factor
        self.upsample2d = Upsample2d(upsampling_factor)
        # Initialize the 2D convolution layer with stride 1
        self.conv2d_stride1 = Conv2d(in_channels, out_channels, kernel_size, stride=1,
                                     weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)

    def forward(self, A):
        # Step 1: Upsample the input
        A_upsampled = self.upsample2d.forward(A)
        # Step 2: Apply convolution with stride 1 to the upsampled input
        Z = self.conv2d_stride1.forward(A_upsampled)
        return Z

    def backward(self, dLdZ):
        # Step 1: Backward pass through the convolution layer
        delta_out = self.conv2d_stride1.backward(dLdZ)
        # Step 2: Backward pass through the upsampling layer
        dLdA = self.upsample2d.backward(delta_out)
        return dLdA

