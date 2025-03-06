import torch
from torch import nn
from layers.multi_head_attn import MultiHeadAttention
from layers.layer_norm import LayerNorm
from layers.ffn import FFN

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8) -> None:
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.FFN = FFN(d_model=d_model, d_ff=d_model*4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        z = self.MHA(x, x, x)
        z = self.dropout1(z)
        z = self.norm1(z + x)
        
        out = self.FFN(z)
        out = self.dropout2(out)
        out = self.norm2(out + z)
        
        return out
        
    