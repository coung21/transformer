import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from embeding.transformer_embeding import TransformerEmbeding

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, enc_vob_size=10000, dec_vob_size=10000, dropout_prob=0.1, device=None):
        super(Transformer, self).__init__()
        self.enc_embed = TransformerEmbeding(d_model, enc_vob_size, max_len=512, device=device)
        self.dec_embed = TransformerEmbeding(d_model, dec_vob_size, max_len=512, device=device)
        
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff, dropout_prob)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff, dropout_prob)
        
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, src, tgt, src_mask, tgt_mask):
        # embeding
        src = self.enc_embed(src)
        tgt = self.dec_embed(tgt)
        
        # encoder and decoder
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # linear and softmax
        output = self.linear(decoder_output)
        output = self.softmax(output)
        return output
    
    def make_causual_mask(self, x, num_heads):
        b, l = x.size()
        mask = torch.triu(torch.ones((l, l), device=x.device), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf')).unsqueeze(0).unsqueeze(0).expand(b, num_heads, l, l)
        return mask
    
    def make_padding_mask(x, num_heads):

        batch_size, seq_len = x.shape

        # Create a mask for <PAD> tokens
        mask = (x != 0).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)

        # Expand the mask to all heads
        mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

        return mask

    
    
        