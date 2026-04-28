# Main Model code for Stable-Diffusion [More advanced then diffusion]

import torch
from torch import nn
from attention import SelfAttention, CrossAttention

## ------- Utility Class -------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, n_embedding:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embedding, 4*n_embedding)
        self.linear_2 = nn.Linear(4*n_embedding, 4*n_embedding)
        
    def forward(self, x:torch.tensor):
        # (1, 320)
        x = self.linear_1(x)
        x = torch.nn.functional.silu(x)
        x = self.linear_2(x)
        
        return x # (1, 1280)


class SwitchSequential(nn.Sequential):
    """ For applying common forward method for all layer
    """
    def forward(self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        ## X is latent here
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
                
        return x
    
    
class upsample(nn.Module):
    """ Bundle a distributive convultion operation with upsampling tensor
    """
    def __init__(self, channels:int):
        super().__init__() 
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (batch_size, Features, H, W) -> (batch_size, Features, H*2, W*2)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

## ---------------------------------------------------

## ------- UNET Layers -------------------------------
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # X: (batch_Size, 320, H/8, W/8)
        x = self.groupnorm(x)
        
        x = nn.functional.silu(x)
        ## Change channel dimension from in_channels to out_channels
        x = self.conv(x)
        # (batch_size, 4, H/8, W/8)
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n_time:int=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Linear layer to get positional factor for time embedddings
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, features, time):
        # Feature: (Batch_size, In_channels, Height, Width)
        # Time: (1, 1280)
        res = features
        
        # Input image - Norm + silu + conv
        features = self.groupnorm_feature(features)
        features = torch.nn.functional.silu(features)
        features = self.conv_feature(features)
        # Time embeddings - silu
        time = nn.functional.silu(time)
        time = self.linear_time(time)
        
        # Combining both to get next layer
        merged = features + time.unsqueeze(-1).unsqueeze(-1) # unsqueeze for [channel,batch] dimensions
        
        merged = self.groupnorm_merged(merged)
        merged = nn.functional.silu(merged)
        merged = self.conv_merged(merged)
        
        
        return merged + self.residual_layer(res)
    

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int, n_embedding:int, d_context:int=768):
        super().__init__()
        channels = n_head * n_embedding
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_cross=d_context, in_proj_bias=False)
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels * 2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, X, context):
        # X: (Batch_size, features, Height, Width)
        # context: (Batch_size, seq_len, Dim)
        res_long = X
        
        X = self.groupnorm(X)
        X = self.conv_input(X)
        
        bz, c, h, w = X.shape
        
        # (Batch_size, features, Height, Width) -> (Batch_size, features, Height * Width)
        X = X.view((bz, c, h*w))
        # (Batch_size, features, Height * Width) -> (Batch_size, Height * Width, features)
        X = X.transpose(-1, -2)
        
        ## Normalization + self_attention with skip connection
        res_short = X
        
        X = self.layernorm_1(X)
        X = self.attention_1(X)
        X = X + res_short
        
        ## Normalization + cross_attention with skip connection
        res_short = X
        
        X = self.layernorm_2(X)
        X = self.attention_2(X, context)
        X = X + res_short
        
        # Final activation using GeGlu == Normalization + FF with GeGlue and skip connections
        res_short = X
        
        X = self.layernorm_3(X)
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        X, gate = self.linear_geglu_1(X).chunk(2, dim=-1)
        X = X * nn.functional.gelu(gate)
        X = self.linear_geglu_2(X)
        X = X + res_short
        
        # Now, reverse any preprocessing
        # (Batch_size, Height * Width, features) -> (Batch_size, features, Height * Width)
        X = X.transpose(-1, -2)
        X = X.view((bz, c, h, w))
        
        X = self.conv_output(X) + res_long
        
        return X
        

class UNET(nn.Module):
    """ Main Model part in Diffusion Architecture for compression and expansion of images """
    def __init__(self,):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            # (Batch_size, 4, H/8, W/8) -> (Batch_size, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8,40)),
            
            # (Batch_size, 320, H/8, W/8) -> (Batch_size, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8,80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8,80)),
            
            # (Batch_size, 640, H/16, W/16) -> (Batch_size, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8,160)),
            
            # (Batch_size, 1280, H/32, W/32) -> (Batch_size, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # (Batch_size, 1280, H/64, W/64) -> (Batch_size, 1280, H/64, W/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )
        
        # Each initial layer input is double to also account for skip connections for encoder
        self.decoders = nn.ModuleList([
            # (B, 2560, H/64, W/64) -> (B, 1280, H/64, W/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), upsample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), 
        ])
        
    def forward(self, latent, context, time):
        # latent: (batch_size, 4, H/8, W/8)
        # context: (batch_size, seq_len , dim)
        # time: (1,1280)
        
        skips = []
        for layers in self.encoders:
            latent = layers(latent, context, time)
            skips.append(latent)
            
        latent = self.bottleneck(latent, context, time)
        
        for layer in self.decoders:
            """ Since we always concat with the skip connections of the encoder,
                the number of features increases before being sent to the decoder's layer
                (i.e. 2560, 1920, 1280, 960, 640, 640, 320, 320)
            """
            skip = skips.pop()
            latent = torch.cat([latent, skip], dim=1) # concat on channel dimension
            latent = layer(latent, context, time)
        
        return latent


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
        
    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor):
        """ Main Class to forward diffusion process
        
        latent: Image which we will process
        context: Textual tensor for context on how to denoise Image
        time: Timestep info for model epoch
        """
        ## Dimensions
        # Latent: (Batch_size, 4, Height/8, Width/8)
        # context: (Batch_size, seq_len, Dim)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)     {1280 = 4*320}
        time = self.time_embedding(time)
        
        # (Batch, 4, H/8, W/8) -> (Batch, 320, H/8, W/8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, H/8, W/8) -> (Batch, 4, H/8, W/8)
        output = self.final(output)

        return output







