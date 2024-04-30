import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # Store the input tensor for use in the backward pass
        self.A = A
        # Unpack the shape of the input tensor
        batch_size, in_channels, input_width = A.shape
        # Calculate the width of the output tensor based on the upsampling factor
        Z = self.upsampling_factor * (input_width - 1) + 1 # TODO
        # Initialize the output tensor with zeros
        Z = np.zeros((batch_size, in_channels, Z))

        # Fill in the values from the input tensor into the output tensor at intervals specified by the upsampling factor
        Z[..., ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        # Calculate the gradient of the loss with respect to the input tensor by sampling from the output gradient
        # at intervals specified by the upsampling factor
        dLdA = dLdZ[..., ::self.upsampling_factor] # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # Unpack the input width from the input tensor's shape and store it for use in the backward pass
        _, _, input_width = A.shape
        self.input_width = input_width

        # Perform the downsampling by selecting elements at intervals specified by the downsampling factor
        Z = A[..., ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Retrieve the input width stored during the forward pass
        input_width = self.input_width
        # Unpack the shape of the output gradient tensor
        batch_size, in_channels, output_width = dLdZ.shape
        # Initialize the gradient tensor with respect to the input as zeros
        dLdA = np.zeros((batch_size, in_channels, input_width))

        # Distribute the gradients from the output back to the appropriate positions in the input tensor,
        # corresponding to the locations sampled during the forward pass
        dLdA[..., ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        # Unpack the shape of the input tensor
        batch_size, in_channels, input_height, input_width = A.shape
        # Calculate the dimensions of the upsampled output tensor
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1
        # Initialize the output tensor with zeros
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        # Fill the output tensor with input values at positions determined by the upsampling factor
        Z[..., ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Extract the gradients corresponding to the input values from the upsampled gradient tensor
        dLdA = dLdZ[..., ::self.upsampling_factor, ::self.upsampling_factor]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # Unpack input tensor dimensions and store input height and width for backward pass
        _, _, input_height, input_width = A.shape
        self.input_height = input_height
        self.input_width = input_width

        # Perform the downsampling by selecting every N-th pixel in both height and width dimensions,
        # where N is the downsampling factor
        Z = A[..., ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Retrieve the batch size and channel dimensions, and the stored input dimensions
        batch_size, in_channels, _, _ = dLdZ.shape
        input_height = self.input_height
        input_width = self.input_width

        # Initialize the gradient with respect to the input tensor as zeros
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))

        # Map the gradients from the downsampled output back to the corresponding pixels in the input tensor
        dLdA[..., ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
