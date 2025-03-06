import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        z = self.linear_1(x)
        z = self.relu(z)
        z = self.linear_2(z)
        
        return z