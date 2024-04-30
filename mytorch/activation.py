import numpy as np


class Identity:

    def forward(self, Z):
        """
        The forward pass for the identity function, which simply returns the input as is.
        
        Arguments:
        Z (np.array): The input tensor.

        Returns:
        np.array: The output tensor, same as the input.
        """
        self.A = Z  # Store the input for use in the backward pass
        return self.A

    def backward(self):
        """
        The backward pass for the identity function, which returns an array of ones since the derivative is 1.

        Returns:
        np.array: The gradient of the identity function with respect to its input.
        """
       
        dAdZ = np.ones(self.A.shape, dtype="f")
        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = np.divide(1,(1 + np.exp(-Z)))
        

        return self.A

    def backward(self):

        dAdZ = self.A - np.multiply(self.A, self.A)

        return dAdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        Z = np.float128(Z) # Use high precision to avoid numerical issues
        self.A = np.divide((np.exp(Z)- np.exp(-Z)), (np.exp(Z) + np.exp(-Z)))
        

        return self.A

    def backward(self):

        dAdZ = 1 - np.multiply(self.A, self.A)

        return dAdZ



class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.A = np.copy(Z)
        self.A [self.A <=0] = 0

        

        return self.A

    def backward(self):
        self.A1 = np.copy(self.A)
        self.A1[self.A1 < 0] = 0
        self.A1[self.A1 > 0 ] = 1
        dAdZ = self.A1
        

        return dAdZ
