
import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]

        se = np.multiply(self.A - self.Y, self.A - self.Y)

        sse =  np.ones((1,self.A.shape[0])) @ (se @ np.ones((self.A.shape[1],1)))
        mse = 0.5 * np.divide(sse, self.N * self.C)

        return mse

    def backward(self):

        dLdA = np.divide((self.A - self.Y), self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]
        
        
        Ones_C = np.ones((self.A.shape[1],1))
        Ones_N = np.ones((self.A.shape[0],1))
        
        exp_A = np.exp(A)
        sumA = np.sum(exp_A, axis = 1, keepdims=True)
 

        self.softmax = np.divide(exp_A, sumA)

        crossentropy = (np.multiply(-self.Y, np.log(self.softmax))) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = self.softmax - self.Y

        return dLdA
