from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter
import torch

def preload_models_from_standard_weights(ckpt_path, device, finetune_weights_path=None):
    # CatVTON parameters
    in_channels = 9
    out_channels = 4

    state_dict=model_converter.load_from_standard_weights(ckpt_path, device)

    diffusion=Diffusion(in_channels=in_channels, out_channels=out_channels).to(device)

    if finetune_weights_path != None:
        checkpoint = torch.load(finetune_weights_path, map_location=device)
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'], strict=True)
    else:
        diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    encoder=VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder=VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    clip=CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        # 'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }