## CLIP Model for Dissusion model -> Convert text to prompt

import torch
from torch import nn

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab:int, n_embedding:int, n_token:int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embedding)
        ## CLIP POS uses parameters instead of sinusodal positional embeddings
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embedding))
        
    def forward(self, tokens):
        # (Batch_size, seq_len) -> (Batch_size, seq_len, Dim)
        X = self.token_embedding(tokens)
        X = X + self.position_embedding
        return X

# Like encoder layer of transformer model
class CLIPLayer(nn.Module):
    """ Layer of Encoding for CLIP Model
        Works just like encoder block from Transformer
    """
    def __init__(self, n_head:int, n_embedding:int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embedding)
        self.layernorm_2 = nn.LayerNorm(n_embedding)
        self.attention = SelfAttention(n_head, n_embedding)
        self.linear_1 = nn.Linear(n_embedding, 4*n_embedding)
        self.linear_2 = nn.Linear(4*n_embedding, n_embedding)
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """ For complex Deep learning architecture, 
            To avoid Exploding gradient and vanishing gradient problem, we normalize before computation layer
        """
        # 1) Optional Self attention + Norm
        res1 = X
        X = self.layernorm_1(X)
        X = self.attention(X, causal_mask=True)
        
        X = X + res1
        
        # 2) Optional linear layer + norm
        res2 = X
        X = self.layernorm_2(X)
        X = self.linear_1(X)
        """ Activation b/w linear layers -> Quick GeLU
            QuickGELU(X) = X*σ(1.702 X)
        """
        X = X*torch.sigmoid(1.702*X)
        X = self.linear_2(X)
        
        X = X + res2
        
        return X
        

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        ## CLIP config is fixed due to us importing it pretrained
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
            
        # (Batch_size, seq_len, dim)
        output = self.layernorm(state)
        
        return output
        
        
        

