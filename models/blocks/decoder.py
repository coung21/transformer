import torch
import torch.nn as nn
from models.layers.multi_head_attn import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.ffn import FFN

class Decoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_prob=0.1):
        super(Decoder, self).__init__()
        self.MHA1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.MHA2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.FFN = FFN(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        
    def forward(self, x, enc_out, causual_mask, padding_mask):
        z1 = self.MHA1(x, x, x, mask=causual_mask)
        z1 = self.dropout1(z1)
        z1 = self.norm1(z1 + x)
        
        z2 = self.MHA2(z1, enc_out, enc_out, mask=padding_mask)
        z2 = self.dropout2(z2)
        z2 = self.norm2(z2 + z1)
        
        out = self.FFN(z2)
        out = self.dropout3(out)
        out = self.norm3(out + z2)
        
        return out          