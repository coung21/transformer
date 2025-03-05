from torch import nn
import math

class TokenEmbeding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbeding, self).__init__(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return super().forward(x) * math.sqrt(self.d_model)