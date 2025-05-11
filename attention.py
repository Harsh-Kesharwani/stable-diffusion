import torch 
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj=nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)

        # This one represents the Wo matrix
        self.out_proj=nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads=n_heads
        self.d_head=d_embed // self.n_heads


    def forward(self, x, causal_mask=False):
        # x: (batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Apply the in_proj to get the queries, keys, and values all at once
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 3 * dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape to (batch_size, seq_len, n_heads, d_head)
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)

        # Transpose for attention dot product: (batch_size, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch_size, n_heads, seq_len, d_head) @ (batch_size, n_heads, d_head, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        attention_weights = q @ k.transpose(-1, -2)

        # Scaling by sqrt(d_head)
        attention_weights = attention_weights / math.sqrt(self.d_head)

        # Causal mask to prevent attending to future tokens
        if causal_mask:
            mask = torch.ones_like(attention_weights, dtype=torch.bool).triu(1)
            attention_weights.masked_fill_(mask, -torch.inf)

        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention weights: (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        output = attention_weights @ v

        # Transpose back: (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)

        # Reshape back to (batch_size, seq_len, dim)
        output = output.reshape(input_shape)

        # Apply output projection
        output = self.out_proj(output)

        return output
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output