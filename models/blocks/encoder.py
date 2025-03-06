import torch
from torch import nn
from layers.multi_head_attn import MultiHeadAttention
from layers.layer_norm import LayerNorm
from layers.ffn import FFN

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8) -> None:
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm = LayerNorm(d_model)
        self.FFN = FFN(d_model=d_model, d_ff=d_model*4)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        z = self.MHA(x, x, x)
        z = self.dropout(z)
        z = self.layer_norm(z + x)
        
        out = self.FFN(z)
        out = self.dropout(out)
        out = self.layer_norm(out + z)
        
        return out
        
    