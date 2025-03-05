from torch import nn
import math

class TokenEmbeding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbeding, self).__init__(vocab_size, d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)