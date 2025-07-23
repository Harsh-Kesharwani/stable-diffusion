from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion, UNET_AttentionBlock
from safetensors.torch import load_file
import model_converter
import torch

def load_finetuned_attention_weights(finetune_weights_path, diffusion, device):
    updated_loaded_data = load_file(finetune_weights_path, device=device)
    print(f"Loaded finetuned weights from {finetune_weights_path}")
    
    unet= diffusion.unet
    idx = 0  
    # Iterate through the attention layers in the encoders
    for layers in unet.encoders:
        for layer in layers:
            if isinstance(layer, UNET_AttentionBlock):
                # Get the parameters from the loaded data for this block
                in_proj_weight_key = f"{idx}.in_proj.weight"
                out_proj_weight_key = f"{idx}.out_proj.weight"
                out_proj_bias_key = f"{idx}.out_proj.bias" 

                # Load the weights if they exist in the loaded data
                if in_proj_weight_key in updated_loaded_data:
                    print(f"Loading {in_proj_weight_key}")
                    layer.attention_1.in_proj.weight.data.copy_(updated_loaded_data[in_proj_weight_key])
                if out_proj_weight_key in updated_loaded_data:
                    print(f"Loading {out_proj_weight_key}")
                    layer.attention_1.out_proj.weight.data.copy_(updated_loaded_data[out_proj_weight_key])
                if out_proj_bias_key in updated_loaded_data:
                    print(f"Loading {out_proj_bias_key}")
                    layer.attention_1.out_proj.bias.data.copy_(updated_loaded_data[out_proj_bias_key])
                    idx += 8

                # Move to the next attention block index in the loaded data


    # Iterate through the attention layers in the decoders
    for layers in unet.decoders:
        for layer in layers:
            if isinstance(layer, UNET_AttentionBlock):
                in_proj_weight_key = f"{idx}.in_proj.weight"
                out_proj_weight_key = f"{idx}.out_proj.weight"
                out_proj_bias_key = f"{idx}.out_proj.bias"

                if in_proj_weight_key in updated_loaded_data:
                    print(f"Loading {in_proj_weight_key}")
                    layer.attention_1.in_proj.weight.data.copy_(updated_loaded_data[in_proj_weight_key])
                if out_proj_weight_key in updated_loaded_data:
                    print(f"Loading {out_proj_weight_key}")
                    layer.attention_1.out_proj.weight.data.copy_(updated_loaded_data[out_proj_weight_key])
                if out_proj_bias_key in updated_loaded_data:
                    print(f"Loading {out_proj_bias_key}")
                    layer.attention_1.out_proj.bias.data.copy_(updated_loaded_data[out_proj_bias_key])
                    idx += 8


    # Iterate through the attention layers in the bottleneck
    for layer in unet.bottleneck:
        if isinstance(layer, UNET_AttentionBlock):
            in_proj_weight_key = f"{idx}.in_proj.weight"
            out_proj_weight_key = f"{idx}.out_proj.weight"
            out_proj_bias_key = f"{idx}.out_proj.bias"

            if in_proj_weight_key in updated_loaded_data:
                print(f"Loading {in_proj_weight_key}")
                layer.attention_1.in_proj.weight.data.copy_(updated_loaded_data[in_proj_weight_key])
            if out_proj_weight_key in updated_loaded_data:
                print(f"Loading {out_proj_weight_key}")
                layer.attention_1.out_proj.weight.data.copy_(updated_loaded_data[out_proj_weight_key])
            if out_proj_bias_key in updated_loaded_data:
                print(f"Loading {out_proj_bias_key}")
                layer.attention_1.out_proj.bias.data.copy_(updated_loaded_data[out_proj_bias_key])
                idx += 8

    print("\nAttention module weights loaded from {finetune_weights_path} successfully.")

def preload_models_from_standard_weights(ckpt_path, device, finetune_weights_path=None):
    # CatVTON parameters
    # in_channels: 8 for instruct-pix2pix (masked free), 9 for sd-v1-5-inpainting (masked based)
    in_channels = 9
    
    if 'maskfree' in finetune_weights_path or 'mask_free' in finetune_weights_path:
        in_channels = 8
        
    out_channels = 4

    state_dict=model_converter.load_from_standard_weights(ckpt_path, device)

    diffusion=Diffusion(in_channels=in_channels, out_channels=out_channels).to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    
    if finetune_weights_path != None:
        load_finetuned_attention_weights(finetune_weights_path, diffusion, device)

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