## Main VAE based decoder block to decode vector into practical images for diffusion model
import torch
from attention import SelfAttention


class VAE_AttentionBlock(torch.nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels) # 1 Head, d_embedding==channels
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channels, H, W)
        res = x
        
        x = self.groupnorm(x)
        
        bz, c, h, w = x.shape
        # (Batch_size, Channels, H, W) -> (Batch_size, Channels, H*W)
        x = x.view((bz, c, h*w))
        # (Batch_size, Channels, H*W) -> (Batch_size, H*W, Channels)
        x = x.transpose(-1, -2)
        
        # Self Attention
        x = self.attention(x)
        
        # (Batch_size, H*W, Channels) -> (Batch_size, Channels, H*W)
        x = x.transpose(-1, -2)
        # (Batch_size, Channels, H*W) -> (Batch_size, Channels, H, W)
        x = x.view((bz, c, h, w))
        
        return x + res


class VAE_ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = torch.nn.GroupNorm(32, in_channels)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = torch.nn.GroupNorm(32, out_channels)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = torch.nn.Identity()
        else:
            self.residual_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Main model is just like residual layer in Transformer
        
        [Norm + Activation function] OR Skip
        """
        # X: (batch_size, input_channels, H, W)
        
        residue = x
        x = self.groupnorm_1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_1(x) # Does not change dimension
        x = self.groupnorm_2(x)
        x = torch.nn.functional.silu(x)
        x = self.conv_2(x) # Does not change dimension
        
        """ Residual can only be added if initial and final dimension are same
            So, We do another convulution to match dimension
        """
        return x + self.residual_layer(residue)


class VAE_Decoder(torch.nn.Sequential):
    """ Return back to original dimension
        Opposite of Encoder
    """
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(4, 4, kernel_size=1, padding=0),
            torch.nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            
            # up.3 (512 -> 512)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # up.2 (512 -> 512)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # up.1 (512 -> 256)
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            # up.0 (256 -> 128)
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            # Normalization and Activation function
            torch.nn.GroupNorm(32, 128),
            torch.nn.SiLU(),
            
            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)     
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # X: (B, 4, H/8, W/8)
        X = X/0.18215 ## Reverse scaling
        
        for module in self:
            X = module(X)
            
        return X # (Batch_size, 3, Height, Width)
    
    