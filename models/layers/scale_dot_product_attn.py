import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None):
        bacth_size, n_head, seq_len, d_k = k.size()
        
        # step 1: dot product
        attn = torch.matmul(q,  k.transpose(-2, -1))
        
        # step 2: scale
        attn = attn / (d_k ** 0.5)
        
        # step 3: mask (opt)
        if mask is not None:
            attn = attn + mask
        
        # step 4: softmax
        score = self.softmax(attn)
        
        # step 5: weighted sum
        output = torch.matmul(score, v)
        
        return output, score

        