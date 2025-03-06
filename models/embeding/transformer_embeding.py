import torch
import torch.nn as nn
from token_embeding import TokenEmbeding
from positional_embeding import PositionalEmbeding

class TransformerEmbeding(nn.Module):
    """
    Transformer Embeding class.
    
    Args:
    vocab_size: int, the size of the vocabulary.
    d_model: int, the dimension of the model.
    max_len: int, the maximum length of the sequence.
    
    Returns:
    torch.Tensor, the transformer embeding shape of (b, seq_len, d_model).
    """
    
    def __init__(self, vocab_size : int, d_model : int, max_len : int, device) -> None:
        super(TransformerEmbeding, self).__init__()
        self.token_embeding = TokenEmbeding(vocab_size, d_model)
        self.positional_embeding = PositionalEmbeding(max_len, d_model, device=device)
        
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return nn.Dropout(p=0.1)(self.token_embeding(x) + self.positional_embeding(x))