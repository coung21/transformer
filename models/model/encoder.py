import torch
import torch.nn as nn
from blocks.encoder import Encoder as EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)])
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    