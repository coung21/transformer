import torch
import torch.nn as nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.embeding.transformer_embeding import TransformerEmbeding

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, enc_vob_size=10000, dec_vob_size=10000, dropout_prob=0.1, device=None):
        super(Transformer, self).__init__()
        self.enc_embed = TransformerEmbeding(d_model=d_model, vocab_size=enc_vob_size, max_len=512, device=device)
        self.dec_embed = TransformerEmbeding(d_model=d_model, vocab_size=dec_vob_size, max_len=512, device=device)
        
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff, dropout_prob)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff, dropout_prob)
        
        
        self.linear = nn.Linear(d_model, dec_vob_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # shared weight between decoder embeding and linear
        self.shared_weight = nn.Parameter(torch.randn(dec_vob_size, d_model))
        nn.init.normal_(self.shared_weight, mean=0.0, std=0.02)
        self.dec_embed.token_embeding.weight = self.shared_weight
        self.linear.weight = self.shared_weight
        
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
    
    def make_padding_mask(self, x, num_heads):

        batch_size, seq_len = x.shape

        # Create a mask for <PAD> tokens
        mask = (x == 1).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        mask = mask.masked_fill(mask == 1, float('-inf'))   

        # Expand the mask to all heads
        mask = mask.expand(batch_size, num_heads, seq_len, seq_len)

        return mask

    
    
        