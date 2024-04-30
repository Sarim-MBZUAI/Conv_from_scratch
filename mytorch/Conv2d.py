import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    # def forward(self, A):
    #     """
    #     Argument:
    #         A (np.array): (batch_size, in_channels, input_height, input_width)
    #     Return:
    #         Z (np.array): (batch_size, out_channels, output_height, output_width)
    #     """
    #     self.A = A

    #     # Z = None  # TODO

    #     # return NotImplemented

    #     batch_size, _, input_height, input_width = A.shape
    #     output_height = input_height - self.kernel_size + 1
    #     output_width = input_width - self.kernel_size + 1

    #     Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

    #     for n in range(batch_size):
    #         for k in range(self.out_channels):
    #             for c in range(self.in_channels):
    #                 for i in range(output_height):
    #                     for j in range(output_width):
    #                         Z[n, k, i, j] += np.sum(A[n, c, i:i+self.kernel_size, j:j+self.kernel_size] * self.W[k, c])
    #             Z[n, k] += self.b[k]

    #     return Z

    # def backward(self, dLdZ):
    #     """
    #     Argument:
    #         dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
    #     Return:
    #         dLdA (np.array): (batch_size, in_channels, input_height, input_width)
    #     """

    #     # self.dLdW = None  # TODO
    #     # self.dLdb = None  # TODO
    #     # dLdA = None  # TODO

    #     # return NotImplemented

    #     batch_size, _, output_height, output_width = dLdZ.shape
    #     _, _, input_height, input_width = self.A.shape

    #     dLdA = np.zeros_like(self.A)
    #     self.dLdW.fill(0)
    #     self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

    #     for n in range(batch_size):
    #         for k in range(self.out_channels):
    #             for c in range(self.in_channels):
    #                 for i in range(output_height):
    #                     for j in range(output_width):
    #                         dLdA[n, c, i:i+self.kernel_size, j:j+self.kernel_size] += dLdZ[n, k, i, j] * self.W[k, c]
    #                         self.dLdW[k, c] += dLdZ[n, k, i, j] * self.A[n, c, i:i+self.kernel_size, j:j+self.kernel_size]

    #     return dLdA

    def forward(self, A):
        self.A = A  # Storing input for use in backward pass
        Z = np.zeros((A.shape[0], self.out_channels, A.shape[2] - self.kernel_size + 1, A.shape[3] - self.kernel_size + 1))

        # Looping over spatial dimensions to apply the convolution
        for i in range(A.shape[2] - self.kernel_size + 1):
            for j in range(A.shape[3] - self.kernel_size + 1):
                # Using tensordot for efficient tensor multiplication, reducing the need for explicit loops over channels
                Z[:, :, i, j] = np.tensordot(A[:, :, i:i+self.kernel_size, j:j+self.kernel_size], self.W, axes=[(1, 2, 3), (1, 2, 3)])

        # Adding bias, broadcasting over batch and spatial dimensions
        Z += self.b.reshape(1, -1, 1, 1)

        return Z

    def backward(self, dLdZ):
        input_size = self.A.shape[2]
        pad_zeros = (self.kernel_size - 1) * 2
        flipped_W = np.flip(self.W, axis=(2, 3))  # Flipping weights for cross-correlation in the backward pass

        dLdA = np.zeros_like(self.A)
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (pad_zeros//2, pad_zeros//2), (pad_zeros//2, pad_zeros//2)), 'constant', constant_values=0)

        # Applying the convolution in the backward pass to compute dLdA
        for i in range(dLdA.shape[2]):
            for j in range(dLdA.shape[3]):
                dLdA[:, :, i, j] = np.tensordot(padded_dLdZ[:, :, i:i+flipped_W.shape[2], j:j+flipped_W.shape[3]], flipped_W, axes=[(1, 2, 3), (0, 2, 3)])

        # Computing dLdW by applying convolution between input activations and gradient of loss w.r.t. output activations
        for i in range(self.dLdW.shape[2]):
            for j in range(self.dLdW.shape[3]):
                self.dLdW[:, :, i, j] = np.tensordot(self.A[:, :, i:i+dLdZ.shape[2], j:j+dLdZ.shape[3]], dLdZ, axes=[(0, 2, 3), (0, 2, 3)]).T

        # Summing over batch and spatial dimensions to compute gradient w.r.t. bias
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA




class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Apply a 2D convolution to the input tensor. This operation is defined in the 'conv2d_stride1' object,
        # which performs the convolution with a stride of 1, maintaining the spatial dimensions of the input.
        Z = self.conv2d_stride1.forward(A)

        # If the stride is greater than 1, apply downsampling to reduce the spatial dimensions of the output.
        # The downsampling operation is encapsulated in the 'downsample2d' object.
        if self.stride > 1:
            Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # If downsampling was applied in the forward pass, compute its backward pass first. This step 'reverses' the
        # downsampling operation, upscaling the gradient to the size it was before downsampling.
        if self.stride > 1:
            dLdZ = self.downsample2d.backward(dLdZ)

        # Compute the backward pass through the 2D convolution layer. This step calculates how much each element
        # of the input tensor contributed to the loss, using the gradient information from the downsampling step if applied.
        dLdA = self.conv2d_stride1.backward(dLdZ)

        return dLdA


