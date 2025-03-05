import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8) -> None:
        super(Encoder, self).__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        z = self.MHA(x, x, x)[0]
        z = nn.Dropout(p=0.1)(z)
        x = self.layer_norm(x + z)
        
        z = self.FFN(x)
        z = nn.Dropout(p=0.1)(z)
        x = self.layer_norm(x + z)
        
        return x
            
    