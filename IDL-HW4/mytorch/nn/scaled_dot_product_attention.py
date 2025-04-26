import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)  # Apply softmax over the source sequence length dimension
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Save inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        # Get key dimension for scaling
        d_k = K.shape[-1]
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, S, E) -> (N, ..., H, L, S)
        # Need to transpose K to align dimensions for matrix multiplication
        K_transposed = np.transpose(K, axes=(*range(K.ndim-2), K.ndim-1, K.ndim-2))  # Transpose last two dimensions
        scaled_dot_product = np.matmul(Q, K_transposed) / np.sqrt(d_k)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S)^T @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev)
        # Transpose attention_scores to swap L and S
        attention_scores_transposed = np.transpose(self.attention_scores, 
                                                  axes=(*range(self.attention_scores.ndim-2), 
                                                       self.attention_scores.ndim-1, 
                                                       self.attention_scores.ndim-2))
        d_V = np.matmul(attention_scores_transposed, d_output)
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, S, Ev)^T -> (N, ..., H, L, S)
        # Transpose V to align dimensions for gradient calculation
        V_transposed = np.transpose(self.V, 
                                   axes=(*range(self.V.ndim-2), 
                                        self.V.ndim-1, 
                                        self.V.ndim-2))
        d_attention_scores = np.matmul(d_output, V_transposed)
        
        # Backprop through softmax
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # If mask was applied, zero out the gradients for masked positions
        if self.mask is not None:
            d_scaled_dot_product = np.where(self.mask, 0, d_scaled_dot_product)
        
        # Scale gradients by sqrt(d_k)
        d_k = self.K.shape[-1]
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        
        # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        # Need to transpose d_scaled_dot_product to align dimensions
        d_scaled_dot_product_transposed = np.transpose(d_scaled_dot_product, 
                                                     axes=(*range(d_scaled_dot_product.ndim-2), 
                                                          d_scaled_dot_product.ndim-1, 
                                                          d_scaled_dot_product.ndim-2))
        d_K = np.matmul(d_scaled_dot_product_transposed, self.Q)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

