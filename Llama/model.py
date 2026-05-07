import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple



@dataclass
class Modelargs:
    """Class with all parameter for Model class"""
    dim:int = 4096
    n_layers:int = 32
    n_heads:int = 32 # Number of heads for the queries
    n_kv_head: Optional[int] = None # Number of heads for Key and Value
    vocab_size:int = -1 # This will be set when we load the tokenizer
    multiple_of:int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps:float = 1e-5
    
    # Needed for KV cache
    max_batch_size:int = 32
    max_seq_len:int = 2048
    
    device:str = None
    
    

def precompute_theta_pos_freq(head_dim:int, seq_len:int, device:str, theta:float=10000.0):
    assert head_dim%2 == 0, "Dimension cannot be odd, must be even numbered"
    # Building the theta parameters -> sequence of frequencies
    """ Formula:
        theta = 10000^(-2[i-1]/dim) for i={1,2,3, ... ,dim/2}
    """
    
    ## shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = (1.0/( theta**(theta_numerator/head_dim) )).to(device)
    
    # Positions for token
    ## shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    ## (seq_len) X (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).to(torch.float32)
    
    ## Compute complex number in the polar form: c=R*exp^[1*m*theta], where R=1,angle=m*theta  as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_pos_emb(q_or_k:torch.Tensor, freqs_complex:torch.Tensor, device:str):
    # step-1: reshape the input tensor to separate the head dimension into two parts
    # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_heads, head_dim/2, 2)
    q_or_k_complex = torch.view_as_complex(q_or_k.view(*q_or_k.shape[:-1], q_or_k.shape[-1]//2, 2).to(torch.float32))
    # Add batch and head dimensions to freqs_complex for broadcasting
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # step-2: apply the rotation by multiplying with the precomputed complex frequencies
    # (batch, seq_len, n_heads, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (batch, seq_len, n_heads, head_dim/2)
    q_or_k_rotated = q_or_k_complex * freqs_complex
    
    # step-3: Seperate the real and imaginary parts and interleave them to get back to the original shape
    # (batch, seq_len, n_heads, head_dim/2, 2) -> (batch, seq_len, n_heads, head_dim)
    q_or_k = torch.view_as_real(q_or_k_rotated).view(*q_or_k.shape)
    return q_or_k.type_as(q_or_k).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        # gamma parameters
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x:torch.Tensor):
        # Compute the root mean square of input along last dimension
        one_over_rms = torch.rsqrt( x.pow(2).mean(-1, keepdim=True)+self.eps )
        return x * one_over_rms

    def forward(self, x:torch.Tensor):
        # (batch, seq_len, dim) -> (batch, seq_len, 1)
        return self._norm(x) * self.weight
    
    
class SelfAttention(nn.Module):
    def __init__(self, args:Modelargs):
        super().__init__()
        self.args = args
        # Number of heads for query
        self.n_q_heads = args.n_heads
        # Number of heads for key and value
        self.n_kv_heads = args.n_kv_head if args.n_kv_head is not None else self.n_q_heads
        # No. of times K,V should be repeated to match the head of query
        self.n_rep = self.n_q_heads // self.n_kv_heads
        
        self.dim = args.dim
        # Dimension for each head
        self.head_dim = self.dim // self.n_q_heads
        
        self.wq = nn.Linear(self.dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, self.dim, bias=False)
        
        self.register_buffer(
            "cache_key",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=torch.float32)
        )
        self.register_buffer(
            "cache_value",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=torch.float32)
        )
        
        
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, Dim)
        # (B, 1, dim) -> (B, 1, N_head_q*head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, N_head_k*head_dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, N_head_v*head_dim)
        xv = self.wv(x)
        
        # (B, 1, N_head_q*head_dim) -> (B, 1, N_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, 1, N_head_kv*head_dim) -> (B, 1, N_head_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        xq = apply_rotary_pos_emb(xq, freqs_complex, device=x.device)
        xk = apply_rotary_pos_emb(xk, freqs_complex, device=x.device)
        
        # Replace the entry in the cache for this token
        self.cache_key[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_value[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # Retrieve all the cached keys and values up to the current position
        # (B, seq_len_curr == start_pos+seq_len)
        k = self.cache_key[:batch_size, :start_pos+seq_len]
        v = self.cache_value[:batch_size, :start_pos+seq_len]
        
        ## Repeat the heads of K,V to reach the number of heads of the queries
        keys = k.repeat_interleave(self.n_rep, dim=2) # (B, seq_len_curr, N_head_q, head_dim)
        values = v.repeat_interleave(self.n_rep, dim=2) # (B, seq_len_curr, N_head_q, head_dim)
        # (batch, 1, N_head_q, head_dim) -> (batch, N_head_q, 1, head_dim)
        xq = xq.transpose(1,2)
        xk = keys.transpose(1,2)
        xv = values.transpose(1,2)
        
        # (batch, N_head_q, 1, head_dim) X (batch, N_head_q, head_dim, seq_len_kv) -> (batch, N_head_q, 1, seq_len_kv)
        scores = torch.matmul(xq, xk.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1).type_as(xq)
        
        # (batch, N_head_q, 1, seq_len_kv) X (batch, N_head_q, seq_len_kv, head_dim) -> (batch, N_head_q, 1, head_dim)
        attention_score = scores.matmul(xv)
        # (batch, N_head_q, 1, head_dim) -> (batch, N_head_q, 1, head_dim) -> (batch, 1, N_head_q*head_dim)
        attention_score = attention_score.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        # (batch, 1, N_head_q*head_dim) -> (batch, 1, dim)
        output = self.wo(attention_score)
        return output
    

class FeedForward(nn.Module):
    def __init__(self, args:Modelargs):
        super().__init__()
        self.dim = args.dim
        
        hidden_dim = int((2/3) * 4 * args.dim)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
            
        self.hidden_dim = ((hidden_dim + args.multiple_of - 1) // args.multiple_of) * args.multiple_of
        
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        
    def forward(self, x:torch.Tensor):
        x1 = self.w1(x)
        x3 = self.w3(x)
        return self.w2(torch.nn.functional.silu(x1) * x3)
        
        

class EncoderBlock(nn.Module):
    def __init__(self, args:Modelargs):
        super().__init__()
        self.args = args
        
        # Attention parameters
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Normalization -> pre attentions + pre feed forward
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.feed_forward_norm = RMSNorm(self.dim, eps=args.norm_eps)
        
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) -> (batch, seq_len, dim)
        attention_out = self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        h = x + attention_out
        
        FFN_out = self.feed_forward.forward(self.feed_forward_norm(h))
        out = h + FFN_out
        
        return out
        

class Transformer(nn.Module):
    """Main Transformer class"""
    def __init__(self, args:Modelargs):
        super().__init__()
        
        assert args.vocab_size != -1, "vocabulary size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_emb = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output_proj = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_freq(self.args.dim // self.args.n_heads, self.args.max_seq_len*2, device=self.args.device)
    
    def forward(self, tokens:torch.Tensor, start_pos:int):
        ## Include seq_len as training needs to be parallelized
        # (batch, seq_len)
        batch_size, seq_len = tokens.shape
        ## Sequence length is always 1 in KV cache
        assert seq_len==1, "Only one token at a time can be processed"
        
        # (batch, seq_len) -> (batch, seq_len, Dim)
        h = self.token_emb(tokens)
        
        # Retrieve the pairs (m,theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        # Consecutively apply all the encoder blocks
        for L in self.layers:
            h = L(h, start_pos, freqs_complex)
            
        h = self.norm(h)
        attention_score = self.output_proj(h).float()
        return attention_score
        
            
        