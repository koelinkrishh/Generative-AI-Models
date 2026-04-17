# %% [markdown]
# # Transformer from Scratch

# %%
import numpy as np
import torch

# %% [markdown]
# ![image](https://media.licdn.com/dms/image/v2/D5612AQHl7OAsf21dnQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1695664075976?e=2147483647&v=beta&t=cSxyV0i3c83zl7agvGhvj7TXcESK1RCMXKNwg5LCqJU)

# %% [markdown]
# ### Note:
# 1. All class will operation on calling forward function [Standard]

# %% [markdown]
# 1. Embedding layer

# %%
class Embeddings(torch.nn.Module):
   def __init__(self, d_model:int, vocab_size:int):
      """ Embedding layer -> map word to vector and train those vector
      d_model: dimension of each word vector
      vocab_size: total number of words in vocabulary
      """
      super().__init__()
      self.d_model = d_model
      self.vocab = vocab_size
      self.embeddings = torch.nn.Embedding(vocab_size, d_model)
             
   def forward(self, x):
      return self.embeddings(x) * np.sqrt(self.d_model)

# %% [markdown]
# 2. Positional Encoding

# %%
class PositionalEncoding(torch.nn.Module):
   def __init__(self, d_model:int, seq_len:int, dropout:float):
      super().__init__()
      self.d_model = d_model
      self.seq_length = seq_len
      self.dropout = torch.nn.Dropout(dropout)
        
      # Create a matrix for positional encoding values
      ## Value -> alternate sinusodal = sin/cos[pos/div_term] = sin/cos[pos*div_term^-1]
      ## 		where div_term = 10000 ^ (2i/d_k)
      pe = torch.zeros(self.seq_length, self.d_model)
        
      position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
      # Using log-space value for numerical stability
      div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        
      # Apply sinusodal to position -> even=sin, odd=cos
      pe[:,0::2] = torch.sin(position*div_term)
      pe[:,1::2] = torch.cos(position*div_term)
        
      # Extend along 3rd dimension for batch
      pe = pe.unsqueeze(0) # (1, seq_length, d_model)        
      
      self.register_buffer('PE', pe) # Save tensor along with code file to not calculate every time
    
   def forward(self, x):
      # Add positional encoding upto current sentence length, all batch, all vector dim
      x = x + (self.PE[:, :x.shape[1], :]) # .requires_grad_(False) -> No need as Pytorch knows PE is not learnable due to buffer
      return self.dropout(x)


# %% [markdown]
# 3. Layer normalization

# %%
# Just normalization the last/innermost layer
class LayerNormalization(torch.nn.Module):
   def __init__(self, features:int, eps:float=1e-9):
      """ Layer Normalization along vector embedding dimension"""
      super().__init__()
      self.eps = eps
      self.mean = torch.nn.Parameter(torch.ones(features))  # Multiple
      self.bias = torch.nn.Parameter(torch.zeros(features)) # Bias
        
   def forward(self, X):
      mean = X.mean(dim=-1, keepdim=True)
      std = X.std(dim=-1, keepdim=True)
      return self.mean* (X-mean) / (std + self.eps) + self.bias

# %% [markdown]
# 4. Feed forward

class FeedForwardBlock(torch.nn.Module):
   def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
      super().__init__()
      # self.inward = torch.nn.Linear(d_model, d_ff)
      # self.activation = torch.nn.LeakyReLU()
      # self.dropout = torch.nn.Dropout(dropout)
      # self.outward = torch.nn.Linear(d_ff, d_model)
        
      self.network = torch.nn.Sequential(
         torch.nn.Linear(d_model, d_ff),
         torch.nn.LeakyReLU(),
         torch.nn.Dropout(dropout),
         torch.nn.Linear(d_ff, d_model),
      )
    
   def forward(self, X):
      return self.network(X)

# %% [markdown]
# 5. Attention layer - Multiple

# %%
class AttentionBlock(torch.nn.Module):
   def __init__(self, d_model:int, h:int, dropout:float):
      super().__init__()
      self.d_model = d_model
      self.h = h
      assert d_model%h==0, "d_model is not divisible by h"        
      self.d_k = d_model//h
      
      # Normally, Weight are matrix but Original paper also include bias
      ## This is cleaner and easier to implement using Linear layer
      self.Wq = torch.nn.Linear(d_model, d_model)
      self.Wk = torch.nn.Linear(d_model, d_model)
      self.Wv = torch.nn.Linear(d_model, d_model)        
      self.Wo = torch.nn.Linear(d_model, d_model)
      self.dropout = torch.nn.Dropout(dropout)   
       
   @staticmethod 
   def attention(q, k, v, dropout:torch.nn.Dropout, mask=None):
      """ Calculate attention score for entire layer        
      return: final result, attention_scores
      """
      d_k = q.shape[-1]
      
      # Key: (Batch, h, seq_len, d_k) -> (Batch, h, d_k, seq_len)
      attention_score = (q@k.transpose(-2,-1))/np.sqrt(d_k) # (Batch, h, seq_len, seq_len)
      
      if mask is not None:
         attention_score.masked_fill_(mask==0, -1e9)
      attention_score = attention_score.softmax(dim=-1) # across last dimension
      if dropout is not None:
         attention_score = dropout(attention_score)        
      return attention_score@v , attention_score
                 
   def forward(self, query, key, value, mask):
      # Each input shape: (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
      q = self.Wq(query)
      k = self.Wk(key)
      v = self.Wv(value)
        
      ## Splitting Each matrix for multiple heads
      # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)  
      q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1,2)
      k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1,2)
      v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1,2)
      # View -> Only change outer shape while preserving inner memory shape
      
      X, atten = AttentionBlock.attention(q, k, v, self.dropout, mask)
      self.attention_scores = atten
        
      # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
      # X = X.transpose(1,2).contiguous().view(X.shape[0], X.shape[1], X.shape[2]*X.shape[3]) OR
      X = X.transpose(1,2).contiguous()
      X = X.view(query.shape[0], query.shape[1], self.h*self.d_k)
      
      output = self.Wo(X) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
      return output


# %%
class ResidualConnection(torch.nn.Module):
   def __init__(self, features: int, dropout:float=0.1):
      super().__init__()
      self.dropout = torch.nn.Dropout(dropout)
      self.norm = LayerNormalization(features)
        
   def forward(self, X, sublayers):
      ## Both implimentation works
      return X + self.dropout(sublayers(self.norm(X)))
      # return X + self.dropout(self.norm(sublayers(X)))

# %% [markdown]
# 6. Residual Connection

# %% [markdown]
# 7. Encoder Block

# %%
class EncoderBlock(torch.nn.Module):
    def __init__(self, features: int, self_atten:AttentionBlock, feed_forward:FeedForwardBlock, dropout:float)->None:
        super().__init__()
        self.attention_block = self_atten
        self.ff = feed_forward
        self.residual_connections = torch.nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # Need two connections
        """ OR
        self.res1 = ResidualConnection(dropout)
        self.res2 = ResidualConnection(dropout)
        """
        
        
    def forward(self, X, mask):
        # Pass sublayer as function for order of execution
        X = self.residual_connections[0](X, sublayers=lambda z: self.attention_block(z,z,z,mask=mask))
        X = self.residual_connections[1](X, sublayers=lambda z: self.ff(z))
        return X


# %% [markdown]
# 8. Encoder

# %%
class Encoder(torch.nn.Module):
   def __init__(self, features: int, layers:torch.nn.ModuleList):
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features)
      
   def forward(self, X, mask):
      for layer in self.layers:
         X = layer(X, mask)
      return self.norm(X)
   

# %% [markdown]
# 9. Decoder Block

# %%
class DecoderBlock(torch.nn.Module):
   def __init__(self, features: int, self_atten:AttentionBlock, cross_atten:AttentionBlock, feed_forward:FeedForwardBlock, dropout:float=0.1):
      super().__init__()
      self.self_attention = self_atten
      self.cross_attention = cross_atten
      self.ff = feed_forward
      self.residual_connections = torch.nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
      
   def forward(self, X, encoder_output, source_mask, target_mask):
      X = self.residual_connections[0](X, lambda z:self.self_attention(z,z,z, target_mask))
      X = self.residual_connections[1](X, lambda z:self.cross_attention(z, encoder_output, encoder_output, source_mask))
      X = self.residual_connections[2](X, self.ff)
      return X
   

# %% [markdown]
# 10. Decoder

# %%
class Decoder(torch.nn.Module):
   def __init__(self, features: int, layers:torch.nn.ModuleList):
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features)
      
   def forward(self, X, encoder_output, source_mask, target_mask):
      for layer in self.layers:
         X = layer(X, encoder_output, source_mask, target_mask)
      return self.norm(X)
   

# %% [markdown]
# 11. Flattening layer

# %%
class ProjectionLayer(torch.nn.Module):
   def __init__(self, d_model:int, vocab_size:int):
      super().__init__()
      self.proj = torch.nn.Linear(d_model, vocab_size)
      
   def forward(self, X):
      # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
      return self.proj(X)


# %% [markdown]
# 12. Transformer Block

# %%
class Transformer(torch.nn.Module):
   def __init__(self, encoder:Encoder, decoder:Decoder,
      source_embed:Embeddings, target_embed:Embeddings, source_pos:PositionalEncoding, target_pos:PositionalEncoding, projection:ProjectionLayer,
      source_vocab_size:int, target_vocab_size:int, d_model:int, dropout:float=0.1):
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.src_embed = source_embed
      self.tgt_embed = target_embed
      self.proj = projection
      self.src_pos = source_pos
      self.tgt_pos = target_pos
      self.input_vocab_size = source_vocab_size
      self.output_vocab_size = target_vocab_size
      self.d_model = d_model
      self.dropout = torch.nn.Dropout(dropout)
      
   def encode(self, source, source_mask):
      # Embedding-> POS-> Encoder-> (self-attention+residue) + (ff+residue)
      source = self.src_embed(source)
      source = self.src_pos(source)
      source = self.encoder(source, source_mask)
      return source
   
   def decode(self, encoder_output, source_mask, target, target_mask):
      # Embedding-> POS-> Decoder-> (masked-self-attention+residue) + (masked-corss-attention+residue) + (ff+residue)
      target = self.tgt_embed(target)
      target = self.tgt_pos(target)
      target = self.decoder(target, encoder_output, source_mask, target_mask)
      return target
      
   def project(self, X):
      # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
      return self.proj(X)

# %%
def build_model(soruce_vocab_size:int, target_vocab_size:int, 
   source_seq_len:int, target_seq_len:int,
   d_model:int, d_ff:int, h:int, num_encoder_layers:int, num_decoder_layers:int, dropout:float=0.1):
   """ Build the complete transformer model with all the layers and blocks
   Parameters:
   soruce_vocab_size: total number of words in source language vocabulary
   target_vocab_size: total number of words in target language vocabulary
   source_seq_len: maximum length of source sentence
   target_seq_len: maximum length of target sentence
   d_model: dimension of word vector
   d_ff: dimension of feed forward network
   h: number of attention heads
   num_encoder_layers: number of encoder blocks
   num_decoder_layers: number of decoder blocks
   dropout: dropout rate for regularization
   """
   
   # Create shared embedding and positional encoding layers
   source_embed = Embeddings(d_model, soruce_vocab_size)
   target_embed = Embeddings(d_model, target_vocab_size)
   # Create the positional encoding layer
   source_pos = PositionalEncoding(d_model, seq_len=source_seq_len, dropout=dropout)
   target_pos = PositionalEncoding(d_model, seq_len=target_seq_len, dropout=dropout)
   
   # Create encoder and decoder blocks
   encoder_layers = torch.nn.ModuleList([EncoderBlock(d_model, AttentionBlock(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(num_encoder_layers)])
   
   decoder_layers = []
   for _ in range(num_decoder_layers):
      decoder_self_atten = AttentionBlock(d_model, h, dropout)
      decoder_cross_atten = AttentionBlock(d_model, h, dropout)
      decoder_ff = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(d_model, decoder_self_atten, decoder_cross_atten, decoder_ff, dropout)
      
      decoder_layers.append(decoder_block)
       
   decoder_layers = torch.nn.ModuleList(decoder_layers)
   
   encoder = Encoder(d_model, encoder_layers)
   decoder = Decoder(d_model, decoder_layers)
   
   projection = ProjectionLayer(d_model, target_vocab_size)
   
   model = Transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection,
      soruce_vocab_size, target_vocab_size, d_model, dropout)
   
   # Initialize weights -> He initialization
   for p in model.parameters():
      if p.dim()>1:
         torch.nn.init.xavier_uniform_(p)
      else:
         torch.nn.init.zeros_(p)
   
   return model



