from torch import nn
import math

class TokenEmbeding(nn.Embedding):
    """
    Token Embeding class.
    
    Args:
    vocab_size: int, the size of the vocabulary.
    d_model: int, the dimension of the model.
    
    Returns:
    torch.Tensor, the token embeding shape of (b, seq_len, d_model).
    """
    
    def __init__(self, vocab_size, d_model):
        super(TokenEmbeding, self).__init__(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return super().forward(x) * math.sqrt(self.d_model)