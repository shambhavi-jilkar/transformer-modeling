import torch.nn as nn
import torch 
from typing import Tuple, Optional

'''
TODO: Implement these Modules.

The file contains three key sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. CrossAttentionLayer: For cross-attention between encoder and decoder
3. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    
    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply self-attention with masking
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights    
    ''' 
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        
        # Initialize the multi-head attention mechanism (use nn.MultiheadAttention)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important: inputs are (batch, seq, features)
        )
        
        # Initialize the normalization layer (use nn.LayerNorm)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (Optional[torch.Tensor]): The padding mask for the key input. shape: (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): The attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # Store residual connection
        residual = x
        
        # Apply pre-normalization
        norm_x = self.norm(x)
        
        # Apply self-attention with masking
        # query, key, and value are all the same for self-attention
        attn_output, mha_attn_weights = self.mha(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Apply residual connection with dropout
        output = residual + self.dropout(attn_output)
        
        # Return the output tensor and attention weights
        return output, mha_attn_weights
    
## -------------------------------------------------------------------------------------------------  
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    This layer is responsible for the cross-attention mechanism between encoder and decoder.
    
    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply cross-attention (query from decoder, key/value from encoder)
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights (both are needed)    
    '''     
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the CrossAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        
        # Initialize the multi-head attention mechanism (use nn.MultiheadAttention)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important: inputs are (batch, seq, features)
        )
        
        # Initialize the normalization layer (use nn.LayerNorm)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x (torch.Tensor): The input tensor from decoder. shape: (batch_size, seq_len, d_model)   
            y (torch.Tensor): The input tensor from encoder. shape: (batch_size, seq_len, d_model)
            key_padding_mask (Optional[torch.Tensor]): The padding mask for the key input. shape: (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): The attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # Store residual connection
        residual = x
        
        # Apply pre-normalization
        norm_x = self.norm(x)
        
        # Apply cross-attention (query from decoder, key/value from encoder)
        attn_output, mha_attn_weights = self.mha(
            query=norm_x,     # Query from decoder
            key=y,            # Key from encoder
            value=y,          # Value from encoder
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Apply residual connection with dropout
        output = residual + self.dropout(attn_output)
        
        # Return the output tensor and attention weights
        return output, mha_attn_weights
    
## -------------------------------------------------------------------------------------------------  
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    
    Steps to implement:
    1. Initialize the feed-forward network as a Sequential with:
       a. First linear layer: d_model -> d_ff
       b. GELU activation
       c. Dropout
       d. Second linear layer: d_ff -> d_model
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply feed-forward network with dropout
       d. Add residual connection
       e. Return the output tensor
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # Initialize the feed-forward network (use nn.Sequential)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Initialize the normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
        ''' 
        # Store residual connection
        residual = x
        
        # Apply pre-normalization
        norm_x = self.norm(x)
        
        # Apply feed-forward network 
        ffn_output = self.ffn(norm_x)
        
        # Apply residual connection with dropout
        output = residual + self.dropout(ffn_output)
        
        # Return the output tensor
        return output

