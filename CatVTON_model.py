
import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm 
from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKL

from utils import (check_inputs_maskfree, get_time_embedding, numpy_to_pil, prepare_image)
from ddpm import DDPMSampler

def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model

    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

class CatVTONPix2PixPipeline:
    def __init__(
        self, 
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=True,
        use_tf32=True,
        models={},
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check
        self.models = models

        self.generator = torch.Generator(device=device)
        self.noise_scheduler = DDPMSampler(generator=self.generator)
        self.encoder= models.get('encoder', None)
        self.decoder= models.get('decoder', None)
        self.unet=models.get('diffusion', None) 
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype) 
        
        # Enable TF32 for faster training on Ampere GPUs
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -1  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image = check_inputs_maskfree(image, condition_image, width, height)
        
        # Ensure consistent dtype for all tensors
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        
        # Encode the image
        image_latent = compute_vae_encodings(image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        
        del image, condition_image
        
        # Concatenate latents
        condition_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
        
        # Prepare noise
        latents = randn_tensor(
            condition_latent_concat.shape,
            generator=generator,
            device=condition_latent_concat.device,
            dtype=self.weight_dtype,
        )
        
        # Prepare timesteps
        self.noise_scheduler.set_inference_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        latents = self.noise_scheduler.add_noise(latents, timesteps[0])
        
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            condition_latent_concat = torch.cat(
                [
                    torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    condition_latent_concat,
                ]
            )

        num_warmup_steps = 0  # For simple DDPM, no warmup needed
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)

                # prepare the input for the inpainting model
                p2p_latent_model_input = torch.cat([latent_model_input, condition_latent_concat], dim=1)
                
                # predict the noise residual
                timestep = t.repeat(p2p_latent_model_input.shape[0])
                time_embedding = get_time_embedding(timestep).to(self.device, dtype=self.weight_dtype)

                noise_pred = self.unet(
                    p2p_latent_model_input,
                    time_embedding
                )
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    t, latents, noise_pred
                )
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        # image = self.decoder(latents.to(self.device, dtype=self.weight_dtype))
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        return image
    