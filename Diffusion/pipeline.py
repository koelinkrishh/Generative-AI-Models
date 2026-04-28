## Main inference pipeline for the diffusion model
import numpy as np
import torch
from tqdm import tqdm
from ddpm import DDPMSampler

Height = 512
Width = 512
Latent_height = Height//8
Latent_width = Width//8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max-new_min)/(old_max-old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freq = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze(-1)*freq[None]
    # Shape: (1, 160*2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    
    


def generate(prompt:str, negative_prompt:str, input_image=None, strength=0.8,
    do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_step=50,
    models={}, seed=None, device=None, idle_device=None, tokenizer=None, precision=torch.float32):
    """
    Generate an image using a diffusion model based on the given prompts.

    This function implements the inference pipeline for a diffusion model, supporting text-to-image generation and image-to-image translation with classifier-free guidance.

    Parameters:
        prompt (str): The positive prompt describing the desired image.
        negative_prompt (str): The negative prompt describing what to avoid in the image.
        input_image (optional): Input image for image-to-image generation. If None, performs text-to-image.
        strength (float): Strength of the input image influence for img2img. Default is 0.8.
        do_cfg (bool): Whether to use classifier-free guidance. Default is True.
        cfg_scale (float): Scale factor for classifier-free guidance. Default is 7.5.
        sampler_name (str): Name of the sampler to use (e.g., 'ddpm'). Default is 'ddpm'.
        n_inference_step (int): Number of inference steps. Default is 50.
        models (dict): Dictionary containing the required models.
        seed (optional): Random seed for reproducibility.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        idle_device: Idle device for offloading models.
        tokenizer: Tokenizer for processing prompts.
        precision: Torch dtype for inference (e.g., torch.float16 or torch.float32).

    Returns:
        The generated image as a numpy array.
    """
    with torch.no_grad():
        if not(0< strength <=1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        ''' Step-1: Generate Random Image via noise  '''
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        ''' Step-2: Generate Encode text for input   '''
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert prompt to text using tokenizer
            pos_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            pos_tokens = torch.tensor(pos_tokens, dtype=torch.long, device=device)
            ## Convert into embedding ->> (batch_size, seq_len, 768=d_embed)
            pos_context = clip(pos_tokens)
            
            ## Also convert negetive string
            neg_tokens = tokenizer.batch_encode_plus([negative_prompt], padding="max_length", max_length=77).input_ids
            neg_tokens = torch.tensor(neg_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, d_embed=768)
            neg_context = clip(neg_tokens)

            # (2, seq_len, dim) -> (2, 77, 768)
            context = torch.cat([pos_context, neg_context])
        else:
            # Convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
            # (1, 77, 768)
            context = clip(tokens)
            
        if idle_device:
            clip.to(idle_device)
        
        if sampler_name=="ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_step(n_inference_step)
        else:
            raise ValueError("Unknown sampler")
        
        latent_shape = (1, 4, Latent_height, Latent_width)
        
        ''' Step-3: Encode input [image,text] '''
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            ''' Preprocessing input '''
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image.resize((Width, Height)))
            input_image_tensor = torch.tensor(input_image_tensor, dtype=precision, device=device)
            
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            # (H, W, Channels) -> (batch_size, H, W, Channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, H, W, Channels) -> (batch_size, Channels, H, W)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)
    
            """ Model uses two noises:
            Encoding Noise: The first instance of sampling noise occurs when using the VAE encoder. The encoder transforms your input image into a latent space representation.
            During this phase, it samples noise from a Gaussian distribution to create a stochastic (probabilistic) encoding. This is a standard part of how a VAE maps an image to a latent distribution,
            ensuring the model can handle variations and generate meaningful latent representations.

            Diffusion Noise: The second instance occurs when the sampler adds noise to those resulting latents to prepare them for the Image-to-Image diffusion process.
            By setting a strength parameter, you decide how much additional noise to inject into the latents. This noise level determines how much of the original image information is obscured;
            the model then uses this "noisy" starting point to iteratively denoise the image based on your prompt.
            """
            ## Random noise
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # Run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)
            
            # (batch_size, 4, Latent_height, Latent_width)
            sampler.set_strength(strength=strength) # Proportion of original image
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            if idle_device:
                encoder.to(idle_device)
        else:
            # Text-to-image => Start from noise
            latents = torch.rand(latent_shape, generator=generator, device=device) # Random noise N(0,1)
        
        
        ''' Step-4: Process input using diffusion model -> Denoise '''
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        
        for i,t in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(t).to(device)
            # (batch_size, 4, Latent_height, Latent_width)
            model_input = latents
            
            if do_cfg:
                # (batch_size, 4, Latent_height, Latent_width) -> (2*batch_size, 4. Latent_height, Latent_width)
                model_input = model_input.repeat(2,1,1,1)
                
            # Model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                output_pos, output_neg = model_output.chunk(2)
                final_output = cfg_scale*(output_pos - output_neg) + output_neg
            else:
                final_output = model_output
                
            ## Remove noise predicted by the UNET
            latents = sampler.step(t, latents, final_output)
            
        if idle_device:
            diffusion.to(idle_device)
        
        ''' Step-5: Decode tensor to get final output image '''
        decoder = models["decoder"]
        decoder.to(device)
        
        images = decoder(latents)
        
        if idle_device:
            decoder.to(idle_device)
        
        ''' Reverse process output '''
        images = rescale(images, (-1,1), (0,255), clamp=True)
        # (batch_size, channel, H, W) -> (batch_size, H, W, channel)
        images = images.permute(0,2,3,1)
        images = images.to("cpu", dtype=torch.uint8).numpy()
        return images[0]
            
            