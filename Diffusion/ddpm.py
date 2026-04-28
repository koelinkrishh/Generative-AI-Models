## Main Schedular/Sampler which will denoise image at each tiemstep
import numpy as np
import torch


class DDPMSampler:
    def __init__(self, generator:torch.Generator, num_training_steps=1000, beta_start:float=0.00085, beta_end:float=0.0120):
        # Params "beta_start" and "beta_end" are taken from pre-trained stable diffusion from CompVis
        ## Linearily spaced variance schedule for noise addition at each timestep
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1.0-self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.one = torch.tensor(1.0)
        self.generator = generator
        
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.arange(num_training_steps-1, -1, -1, dtype=torch.long)

    def set_inference_step(self, num_inference_steps:int=50):
        self.num_inference_timesteps = num_inference_steps
        self.step_ratio = self.num_train_timesteps//self.num_inference_timesteps
        timesteps = torch.arange(0, num_inference_steps, dtype=torch.int64) * self.step_ratio
        self.timesteps = torch.flip(timesteps, dims=(0,))
    
    def _get_previous_timesteps(self, timestep:int)->int:
        prev_t = timestep - self.step_ratio
        return prev_t
    
    def _get_variance(self, timestep:int)->int:
        prev_t = self._get_previous_timesteps(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_minus_1 = self.alpha_cumprod[prev_t] if prev_t>=0 else self.one
        beta_t = 1-alpha_prod_t/alpha_prod_t_minus_1
        
        # for t>0, compute predicted variance βt and sample from it to get previous sample
        ## Fix variance sigma_t = sqrt(βt) and add noise to mean for sampling
        variance = (1-alpha_prod_t_minus_1)/(1-alpha_prod_t)*beta_t
        variance = variance.clamp(min=1e-12)
        return variance
    
    def set_strength(self, strength=1):
        """ Function to set how much noise to add to the input image
            More noise[strength~1] means more influence of the input image
            less noise[strength~0] means less influence of the input image.
        """
        # start_step is the number of noise levels to skip
        self.start_step = self.num_inference_timesteps - int(self.num_inference_timesteps*strength)
        self.timesteps = self.timesteps[self.start_step:]
        
    def step(self, timestep:int, latents:torch.Tensor, model_output:torch.Tensor):
        """ Function to compute Mean of guassian for noise removal at each timestep.
            Variance is computed separately and added as noise to the mean for sampling.
            
        timestep: t -> to get current timestep coefficients
        latents: x -> Actual image at current timestep
        model_output: epision -> error predicted by model
        """
        prev_t = self._get_previous_timesteps(timestep)
        
        # 1. Compute coefficients
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_minus_1 = self.alpha_cumprod[prev_t] if prev_t>=0 else self.one
        beta_prod_t = 1-alpha_prod_t
        beta_prod_t_minus_1 = 1-alpha_prod_t_minus_1
        current_alpha_t = alpha_prod_t/alpha_prod_t_minus_1
        current_beta_t = 1-current_alpha_t
        
        # 2. Compute predicted original sample
        pred_original_sample = (latents - torch.sqrt(beta_prod_t)*model_output)/ torch.sqrt(alpha_prod_t)
        
        # 3. Compute coefficient for the pred_original_sample x0 and current sample xt
        pred_orignal_sample_coeff = (torch.sqrt(alpha_prod_t_minus_1)*current_beta_t)/beta_prod_t
        current_sample_coeff = (torch.sqrt(current_alpha_t)*beta_prod_t_minus_1)/beta_prod_t
        
        # 4. Compute predicted sample µt
        pred_sample_mean = pred_orignal_sample_coeff*pred_original_sample + current_sample_coeff*latents
        # 5. Compute variance for noise addition
        if timestep>0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = noise*torch.sqrt(self._get_variance(timestep))
        else:
            variance = 0.0
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        pred_prev_sample = pred_sample_mean + variance
        
        return pred_prev_sample
        
        
    def add_noise(self, original_sample:torch.FloatTensor, timesteps:torch.LongTensor):
        alpha_cumprod = self.alpha_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)
        
        sqrt_alpha_prod = ((alpha_cumprod[timesteps])**0.5)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = (1-alpha_cumprod[timesteps])**0.5 ## Standard deviation of noise
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, dtype=original_sample.dtype)
        ## Sqrt of 2nd term for getting stadnard deviation. Equation states variance = 1 - alpha_prod
        noisy_samples = sqrt_alpha_prod*original_sample + sqrt_one_minus_alpha_prod*noise
        
        return noisy_samples
