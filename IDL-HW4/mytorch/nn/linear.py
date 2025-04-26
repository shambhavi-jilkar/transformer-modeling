import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # Store input for backward pass
        self.A = A
        
        # Perform linear transformation: Z = AW^T + b
        # Reshape A to 2D if it's not already
        original_shape = A.shape
        in_features = self.W.shape[1]
        out_features = self.W.shape[0]
        
        # Reshape A to (batch_size, in_features) where batch_size is the product of all dimensions except the last
        reshaped_A = A.reshape(-1, in_features)
        
        # Compute linear transformation
        Z_reshaped = np.matmul(reshaped_A, self.W.T) + self.b
        
        # Reshape Z back to match the input shape but with the last dimension replaced with out_features
        new_shape = original_shape[:-1] + (out_features,)
        Z = Z_reshaped.reshape(new_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Get shapes
        original_shape = dLdZ.shape
        out_features = self.W.shape[0]
        in_features = self.W.shape[1]
        
        # Reshape dLdZ to 2D
        reshaped_dLdZ = dLdZ.reshape(-1, out_features)
        reshaped_A = self.A.reshape(-1, in_features)
        
        # Compute gradients
        # dLdA = dLdZ * W
        self.dLdA = np.matmul(reshaped_dLdZ, self.W)
        
        # dLdW = dLdZ^T * A
        self.dLdW = np.matmul(reshaped_dLdZ.T, reshaped_A)
        
        # dLdb = sum(dLdZ, axis=0)
        self.dLdb = np.sum(reshaped_dLdZ, axis=0)
        
        # Reshape dLdA back to match the input shape
        dLdA_reshaped = self.dLdA.reshape(self.A.shape)
        
        # Return gradient of loss wrt input
        return dLdA_reshaped
