import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # Get batch size and sequence length
    N = padded_input.shape[0]
    T = padded_input.shape[1]
    
    # Create a range tensor representing positions
    positions = torch.arange(0, T, device=padded_input.device).expand(N, T)
    
    # Create the mask: position >= length means it's a padding position
    # Expand input_lengths to shape (N, 1) for broadcasting
    mask = positions >= input_lengths.unsqueeze(1)
    
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # Get sequence length
    T = padded_input.shape[1]
    
    # Create mask: upper triangular where each position can only attend to itself and previous tokens
    # tril returns lower triangular part (including diagonal)
    # We negate it to get the upper triangular part (excluding diagonal)
    # This gives us positions that should be masked (marked as True)
    mask = ~torch.tril(torch.ones(T, T, device=padded_input.device, dtype=torch.bool))
    
    return mask

