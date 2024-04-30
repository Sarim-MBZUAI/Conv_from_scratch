import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        # Store the original shape of the input tensor for use in the backward pass
        self.A = A
        # Reshape the input tensor to a 2D tensor, where the first dimension is the batch size,
        # and the second dimension is the product of the other dimensions of the input tensor.
        Z = A.reshape(A.shape[0], A.shape[1] * A.shape[2])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        # Reshape the gradient tensor back to the original shape of the input tensor.
        # This is necessary because the gradient tensor needs to match the shape of the input tensor for correct gradient propagation.
        dLdA = dLdZ.reshape(self.A.shape[0], self.A.shape[1], self.A.shape[2])

        return dLdA
