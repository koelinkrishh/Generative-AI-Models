## Main VAE based encoder block to encode input image into semantic vector for diffusion model
import torch
from decoder import VAE_AttentionBlock, VAE_ResidualBlock



class VAE_Encoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, H, W)
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1), # [3 channel to 128 channel]
            
            #  (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            #  (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_size, 128, H, W) -> (Batch_size, 128, H/2, W/2)
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_size, 256, H/2, W/2) -> (Batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, 256, H/4, W/4) -> (Batch_size, 256, H/4, W/4)
            torch.nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),
            
            # (Batch_size, 256, H/4, W/4) -> (Batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_size, 512, H/4, W/4) -> (Batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, H/4, W/4) -> (Batch_size, 512, H/8, W/8)
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            
            ## Now we normalize our image values
            torch.nn.GroupNorm(32,512),
            
            ## Activation function -> SiLU -> Silu(X) = X*sigmoid(X)
            torch.nn.SiLU(),
            
            # (Batch_size, 512, H/8, W/8) -> (Batch_size, 8, H/8, W/8)
            torch.nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (Batch_size, 8, H/8, W/8) -> (Batch_size, 8, H/8, W/8)
            torch.nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )
    
    def forward(self, x:torch.Tensor, noise:torch.Tensor)-> torch.Tensor:
        """ Encoding an Image into normal distribution
            
            X gives us that distributions parameters -> mean,var => q(z∣x)=N(μ(x),σ2(x))
        """
        # Input: (Batch_Size, Channel, Height, Width)
        # Noise: (Batch_size, Out_channel, Height/8, Width/8)
        
        # We need to apply special padding for conv layer with stride
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # Pad on right and bottom only
                x = torch.nn.functional.pad(x, [0,1,0,1]) # (left, right, top, bottom)
            x = module(x)
        
        # (Batch_size, 8, H/8, W/8) -> two tensor of shape (batch_size, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, chunks=2, dim=1) # split into two chunks along dim=1

        """ Instead of var, train model to predict log(var). More stable and easier """
        log_var = torch.clamp(log_var, -30, 20) # clamp for numerical stability
        variance = log_var.exp()
        std_dev = variance.sqrt() # (batch_size, 4, H/8, W/8)
        
        # Now, we will sample a random noise { N[0,1] }
        # Z=N[0,1] -> N[mean, variance]=X --> X = mean + std_dev*Z
        """ Reparameterization Trick: z=μ+σ⋅ϵ, ϵ∼N(0,1) """
        x = mean + std_dev*noise 
        
        # Scale the output by constant - empirical scaling factor to scale latent space
        ## 0.18215 ≈ 1/(std of latent space)
        x = x*0.18215
        
        return x




