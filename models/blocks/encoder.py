import torch
from torch import nn
from models.layers.multi_head_attn import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.ffn import FFN

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_probs=0.1):
        super(Encoder, self).__init__()
        self.MHA = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.FFN = FFN(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(dropout_probs)
        self.dropout2 = nn.Dropout(dropout_probs)
        
    def forward(self, x, padding_mask):
        z = self.MHA(x, x, x, mask=padding_mask)
        z = self.dropout1(z)
        z = self.norm1(z + x)
        
        out = self.FFN(z)
        out = self.dropout2(out)
        out = self.norm2(out + z)
        
        return out
        
    