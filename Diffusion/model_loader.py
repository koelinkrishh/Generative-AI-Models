from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter
import torch

def preload_models_from_standard_weights(ckpt_path, device, precision=torch.float32):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    
    encoder_model = VAE_Encoder().to(device, dtype=precision)
    encoder_model.load_state_dict(state_dict['encoder'], strict=True)
    
    decoder_model = VAE_Decoder().to(device, dtype=precision)
    decoder_model.load_state_dict(state_dict['decoder'], strict=True)
    
    Diffusion_model = Diffusion().to(device, dtype=precision)
    Diffusion_model.load_state_dict(state_dict['diffusion'], strict=True)
    
    CLIP_model = CLIP().to(device, dtype=precision)
    CLIP_model.load_state_dict(state_dict['clip'], strict=True)
    
    return {'clip': CLIP_model, 'encoder': encoder_model, 'decoder': decoder_model, 'diffusion': Diffusion_model}
