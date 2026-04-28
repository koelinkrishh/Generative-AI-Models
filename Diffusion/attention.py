# Attention block for diffusion model
import torch
from torch import nn
import math



class SelfAttention(nn.Module):
    def __init__(self, n_head:int, d_embed:int, in_proj_bias:bool=True, out_proj_bias:bool=True):
       super().__init__()
       # 3 times width for Query,Key and Value
       self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
       self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
       
       self.n_heads = n_head
       self.dimension_head = d_embed//n_head
       
    def forward(self, X:torch.Tensor, causal_mask=False):
        # X: (batch_size, seq_len, Dim)
        input_shape = X.shape
        batch_size, seq_len, d_emb = input_shape
        
        intermediate_shape = (batch_size, seq_len, self.n_heads, self.dimension_head)
        
        # (B, seq_len, dim) -> (B, seq_len, dim*3) -> 3 tensor of (B, seq_len, dim)
        q, k, v = self.in_proj(X).chunk(3, dim=-1)

        # (B, seq_len, dim) -> (B, seq_len, Heads, Dim/Heads)
        q = q.view(intermediate_shape).transpose(1,2)
        k = k.view(intermediate_shape).transpose(1,2)
        v = v.view(intermediate_shape).transpose(1,2)
        """ Alternative
        weights = q@k.transpose(-1,-2)
        
        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, -torch.inf) # fill with -ve infinity
        
        weights = nn.functional.softmax(weights/math.sqrt(self.dimension_head), dim=-1)
        
        # (B, Head, seq_len, seq_len) @ (B, Head, seq_len, Dim/Heads) -> (B, Head, seq_len, Dim/Heads)
        output = weights@v
        """
        # Using optimized scaled_dot_product_attention
        output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal_mask)
        
        # (B, head, seq_len, Dim/heads) -> (B, seq_len, head, Dim/head) -> (B, seq_len, Dim)
        output = output.transpose(1, 2).reshape(input_shape)
        ## Project through Wo
        output = self.out_proj(output)
        
        # (B, seq_len, Dim)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, n_head:int, n_embedding:int, d_cross:int, in_proj_bias:bool=True, out_proj_bias:bool=True):
        super().__init__()
        self.q_proj = nn.Linear(n_embedding, n_embedding, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, n_embedding, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, n_embedding, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(n_embedding, n_embedding, bias=out_proj_bias)
        
        self.n_head = n_head
        self.d_head = n_embedding//n_head
        
    def forward(self, x, y):
        # x: [latent] - (Batch_size, seq_len_query, dim_query)
        # y: [context] - (Batch_size, seq_len_key_value, dim_kv)
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        interrim_shape = (batch_size, -1, self.n_head, self.d_head)
        
        # (Batch_size, seq_len**, dim_query)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # (batch_size, seq_len, head_count, dim) -> (batch_size, head_count, seq_len, dim)
        q = q.view(interrim_shape).transpose(1,2)
        k = k.view(interrim_shape).transpose(1,2)
        v = v.view(interrim_shape).transpose(1,2)
        """ # Alternative
        attention_weight = (q@k.transpose(-1,-2))/math.sqrt(self.d_head)
        ## No need to apply causal mask
        attention_weight = nn.functional.softmax(attention_weight, dim=-1)
        
        output = attention_weight@v
        """
        # Using optimized scaled_dot_product_attention
        output = nn.functional.scaled_dot_product_attention(q, k, v)
        
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        
        return self.out_proj(output)
        
        
        
    
    
    
    
    
    
    
