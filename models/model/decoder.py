import torch
import torch.nn as nn
from models.blocks.decoder import Decoder as DecoderLayer

class Decoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout_prob=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_layers)])
        
    def forward(self, x, encoder_output, causual_mask, padding_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, causual_mask, padding_mask)
        return x