import torch
import torch.nn as nn
from scale_dot_product_attn import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
         # step 1: linear projection
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # step 2: split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # step 3: scale dot product attention
        output, score = self.attention(q, k, v, mask)
        
        # step 4: concat heads
        output = self.concat_heads(output)
        
        # step 5: final linear projection
        output = self.w_concat(output)
        
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # divide d_model into n_head * d_k
        d_head = d_model // self.num_heads
        
        # split into num_heads
        x = x.view(batch_size, seq_len, self.num_heads, d_head).permute(0, 2, 1, 3)
        
        return x
    
    def concat_heads(self, x):
        batch_size, n_head, seq_len, d_head = x.size()
        
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_head * n_head)

        return x