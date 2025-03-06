import torch
from torch import nn

class PositionalEmbeding(nn.Module):
    """
    Positional Embeding class.
    
    Args:
    max_len: int, the maximum length of the sequence.
    d_model: int, the dimension of the model.
    
    Returns:
    torch.Tensor, the positional embeding shape of (b, seq_len, d_model).
    """
    def __init__(self, max_len : int, d_model : int, device):
        super(PositionalEmbeding, self).__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model, device=device).float()
        self.pe.requires_grad = False
        
        pos = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        div_term = torch.exp(10000, _2i / d_model)
        
        self.pe[:, 0::2] = torch.sin(pos / div_term)
        self.pe[:, 1::2] = torch.cos(pos / div_term)
    
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        return self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)