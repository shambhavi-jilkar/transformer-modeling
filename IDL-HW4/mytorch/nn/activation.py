import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # Subtract max for numerical stability
        shifted_Z = Z - np.max(Z, axis=self.dim, keepdims=True)
        # Exponentiate
        exp_Z = np.exp(shifted_Z)
        # Normalize
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Initialize the output gradient with zeros
        dLdZ = np.zeros_like(dLdA)
        
        # Convert negative dim to positive for easier handling
        pos_dim = self.dim if self.dim >= 0 else len(self.A.shape) + self.dim
        
        # Get the shape of input
        shape = self.A.shape
        
        # Use a simpler approach based on the formula:
        # dLdZ_i = sum_j(dLdA_j * (delta_ij - A_j) * A_i)
        # where delta_ij is 1 if i=j, 0 otherwise
        
        # Handle different dimensions by reshaping and using batch operations
        if len(shape) == 1:
            # For 1D case, we can directly use the formula
            for i in range(shape[0]):
                for j in range(shape[0]):
                    if i == j:
                        dLdZ[i] += dLdA[j] * self.A[i] * (1 - self.A[j])
                    else:
                        dLdZ[i] += dLdA[j] * (-self.A[i] * self.A[j])
        
        elif len(shape) == 2:
            # For 2D case (batch, features)
            batch_size, num_features = shape
            
            if pos_dim == 1:  # Softmax along feature dimension
                for b in range(batch_size):
                    for i in range(num_features):
                        for j in range(num_features):
                            if i == j:
                                dLdZ[b, i] += dLdA[b, j] * self.A[b, i] * (1 - self.A[b, j])
                            else:
                                dLdZ[b, i] += dLdA[b, j] * (-self.A[b, i] * self.A[b, j])
            else:  # Softmax along batch dimension
                for f in range(num_features):
                    for i in range(batch_size):
                        for j in range(batch_size):
                            if i == j:
                                dLdZ[i, f] += dLdA[j, f] * self.A[i, f] * (1 - self.A[j, f])
                            else:
                                dLdZ[i, f] += dLdA[j, f] * (-self.A[i, f] * self.A[j, f])
                                
        elif len(shape) == 3:
            # For 3D case (batch, seq_len, features)
            batch_size, seq_len, num_features = shape
            
            if pos_dim == 2:  # Softmax along feature dimension
                for b in range(batch_size):
                    for s in range(seq_len):
                        for i in range(num_features):
                            for j in range(num_features):
                                if i == j:
                                    dLdZ[b, s, i] += dLdA[b, s, j] * self.A[b, s, i] * (1 - self.A[b, s, j])
                                else:
                                    dLdZ[b, s, i] += dLdA[b, s, j] * (-self.A[b, s, i] * self.A[b, s, j])
            elif pos_dim == 1:  # Softmax along sequence length
                for b in range(batch_size):
                    for f in range(num_features):
                        for i in range(seq_len):
                            for j in range(seq_len):
                                if i == j:
                                    dLdZ[b, i, f] += dLdA[b, j, f] * self.A[b, i, f] * (1 - self.A[b, j, f])
                                else:
                                    dLdZ[b, i, f] += dLdA[b, j, f] * (-self.A[b, i, f] * self.A[b, j, f])
            else:  # Softmax along batch dimension
                for s in range(seq_len):
                    for f in range(num_features):
                        for i in range(batch_size):
                            for j in range(batch_size):
                                if i == j:
                                    dLdZ[i, s, f] += dLdA[j, s, f] * self.A[i, s, f] * (1 - self.A[j, s, f])
                                else:
                                    dLdZ[i, s, f] += dLdA[j, s, f] * (-self.A[i, s, f] * self.A[j, s, f])
        else:
            # For higher dimensions, we can use a more general approach
            # This is a simplified approach for the most common case: softmax along the last dimension
            if pos_dim == len(shape) - 1:
                # For each element, compute the gradient
                # First, reshape to collapse all dimensions except the softmax dimension
                reshaped_A = self.A.reshape(-1, shape[pos_dim])
                reshaped_dLdA = dLdA.reshape(-1, shape[pos_dim])
                reshaped_dLdZ = np.zeros_like(reshaped_dLdA)
                
                # Compute gradients for each batch
                for b in range(reshaped_A.shape[0]):
                    # Compute the Jacobian-vector product
                    jacobian = np.diag(reshaped_A[b]) - np.outer(reshaped_A[b], reshaped_A[b])
                    reshaped_dLdZ[b] = np.dot(jacobian, reshaped_dLdA[b])
                
                # Reshape back to original shape
                dLdZ = reshaped_dLdZ.reshape(shape)
            else:
                # For other dimensions, we would need a more complex implementation
                # This is a placeholder to indicate that it's not supported in this simplified version
                raise NotImplementedError("Softmax backward for arbitrary dimensions not fully implemented.")
                
        return dLdZ


