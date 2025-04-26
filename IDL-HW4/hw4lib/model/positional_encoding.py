import torch
from torch import nn
import math

'''
TODO: Implement this Module.

Specification:
- Module should add positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix should have shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values should be on same device as input tensor
- Should handle any sequence length up to max_len
- Should raise error if input sequence length exceeds max_len
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Side Effects:
            - Initializes the positional encoding buffer 'pe' 
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create position tensor: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for wavelength calculation
        # For even indices: sin(pos / 10000^(2i/d_model))
        # For odd indices: cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (if dimension is odd, handle the last element)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # Get sequence length from input tensor
        seq_len = x.size(1)
        
        # Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        
        # Add positional encodings to input
        # Take only the needed sequence length from the pre-computed encodings
        x = x + self.pe[:, :seq_len, :]
        
        return x
