import os
import torchvision.transforms as transforms
from PIL import Image
import math
import PIL
import numpy as np
import torch
from PIL import Image
from accelerate.state import AcceleratorState
from packaging import version
import accelerate
from typing import List, Optional, Tuple, Set
# from diffusers import UNet2DConditionModel, SchedulerMixin
from tqdm import tqdm
from PIL import Image, ImageFilter

def get_time_embedding(timesteps):
    # Handle both scalar and batch inputs
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (B, 160)
    x = timesteps.float()[:, None] * freqs[None]
    # Shape: (B, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# Prepare the input for inpainting model.
def prepare_inpainting_input(
    noisy_latents: torch.Tensor, 
    mask_latents: torch.Tensor,
    condition_latents: torch.Tensor,
    enable_condition_noise: bool = True,
    condition_concat_dim: int = -1,
) -> torch.Tensor:
    """
    Prepare the input for inpainting model.
    
    Args:
        noisy_latents (torch.Tensor): Noisy latents.
        mask_latents (torch.Tensor): Mask latents.
        condition_latents (torch.Tensor): Condition latents.
        enable_condition_noise (bool): Enable condition noise.
    
    Returns:
        torch.Tensor: Inpainting input.
    """
    if not enable_condition_noise:
        condition_latents_ = condition_latents.chunk(2, dim=condition_concat_dim)[-1]
        noisy_latents = torch.cat([noisy_latents, condition_latents_], dim=condition_concat_dim)
    noisy_latents = torch.cat([noisy_latents, mask_latents, condition_latents], dim=1)
    return noisy_latents

# Compute VAE encodings
def compute_vae_encodings(image_tensor, encoder, device="cuda"):
    """Encode image using VAE encoder"""
    # Generate random noise for encoding
    encoder_noise = torch.randn(
        (image_tensor.shape[0], 4, image_tensor.shape[2] // 8, image_tensor.shape[3] // 8),
        device=device,
    )
    
    # Encode using your custom encoder
    latent = encoder(image_tensor, encoder_noise)
    return latent


def check_inputs(image, condition_image, mask, width, height):
    if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
        return image, condition_image, mask
    assert image.size == mask.size, "Image and mask must have the same size"
    image = resize_and_crop(image, (width, height))
    mask = resize_and_crop(mask, (width, height))
    condition_image = resize_and_padding(condition_image, (width, height))
    return image, condition_image, mask


def repaint_result(result, person_image, mask_image):
    result, person, mask = np.array(result), np.array(person_image), np.array(mask_image)
    # expand the mask to 3 channels & to 0~1
    mask = np.expand_dims(mask, axis=2)
    mask = mask / 255.0
    # mask for result, ~mask for person
    result_ = result * mask + person * (1 - mask)
    return Image.fromarray(result_.astype(np.uint8))


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            # Batch and add channel dim for single mask
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            # Single mask, the 0'th dimension is considered to be
            # the existing batch size of 1
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            # Batch of mask, the 0'th dimension is considered to be
            # the batching dimension
            mask_image = mask_image.unsqueeze(1)

        # Binarize mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]

        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

    return mask_image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def tensor_to_image(tensor: torch.Tensor):
    """
    Converts a torch tensor to PIL Image.
    """
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Input tensor should be in range [0, 1]."
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image

def resize_and_crop(image, size):
    # Crop to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(
        ((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2)
    )
    # resize
    image = image.resize(size, Image.LANCZOS)
    return image


def resize_and_padding(image, size):
    # Padding to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # padding
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding

def save_debug_visualization(
    person_images, cloth_images, masks, masked_image, 
    noisy_latents, predicted_noise, target_latents, 
    decoder, global_step, output_dir, device="cuda"
):
    """
    Simple debug visualization function to save training progress images.
    
    Args:
        person_images: Original person images [B, 3, H, W]
        cloth_images: Cloth/garment images [B, 3, H, W]  
        masks: Mask images [B, 1, H, W]
        masked_image: Person image with mask applied [B, 3, H, W]
        noisy_latents: Noisy latents fed to model [B, C, h, w]
        predicted_noise: Model's predicted noise [B, C, h, w]
        target_latents: Ground truth latents [B, C, h, w]
        decoder: VAE decoder model
        global_step: Current training step
        output_dir: Directory to save images
        device: Device to use
    """
    
    try:
        with torch.no_grad():
            # Take first sample from batch
            person_img = person_images[0:1]  # [1, 3, H, W]
            cloth_img = cloth_images[0:1]
            mask_img = masks[0:1] 
            masked_img = masked_image[0:1]
            
            # Split concatenated latents if needed (assuming concat on height dim)
            if target_latents.shape[-2] > noisy_latents.shape[-2] // 2:
                # Latents are concatenated, split them
                h = target_latents.shape[-2] // 2
                noisy_person_latent = noisy_latents[0:1, :, :h, :]
                predicted_person_latent = (noisy_person_latent - predicted_noise[0:1, :, :h, :])
                target_person_latent = target_latents[0:1, :, :h, :]
            else:
                noisy_person_latent = noisy_latents[0:1]
                predicted_person_latent = (noisy_person_latent - predicted_noise[0:1])
                target_person_latent = target_latents[0:1]
            
        
            # Decode latents to images
            with torch.cuda.amp.autocast(enabled=False):
                noisy_decoded = decoder(noisy_person_latent.float())
                predicted_decoded = decoder(predicted_person_latent.float()) 
                target_decoded = decoder(target_person_latent.float())
            
            # Convert to PIL images
            def tensor_to_pil(tensor):
                # tensor: [1, 3, H, W] in range [-1, 1] or [0, 1]
                tensor = tensor.squeeze(0)  # [3, H, W]
                tensor = torch.clamp((tensor + 1.0) / 2.0, 0, 1)  # Normalize to [0,1] 
                tensor = tensor.cpu()
                transform = transforms.ToPILImage()
                return transform(tensor)
            
            # Convert mask to PIL (single channel)
            def mask_to_pil(tensor):
                tensor = tensor.squeeze()  # Remove batch and channel dims
                tensor = torch.clamp(tensor, 0, 1)
                tensor = tensor.cpu()
                # Convert to 3-channel for visualization
                tensor_3ch = tensor.unsqueeze(0).repeat(3, 1, 1)
                transform = transforms.ToPILImage()
                return transform(tensor_3ch)
            
            # Convert all tensors to PIL images
            person_pil = tensor_to_pil(person_img)
            cloth_pil = tensor_to_pil(cloth_img) 
            mask_pil = mask_to_pil(mask_img)
            masked_pil = tensor_to_pil(masked_img)
            noisy_pil = tensor_to_pil(noisy_decoded)
            predicted_pil = tensor_to_pil(predicted_decoded)
            target_pil = tensor_to_pil(target_decoded)
            
            # Create labels
            labels = ['Person', 'Cloth', 'Mask', 'Masked', 'Noisy', 'Predicted', 'Target']
            images = [person_pil, cloth_pil, mask_pil, masked_pil, noisy_pil, predicted_pil, target_pil]
            
            # Get dimensions
            width, height = person_pil.size
            
            # Create combined image (horizontal layout)
            combined_width = width * len(images)
            combined_height = height + 30  # Extra space for labels
            
            combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
            
            # Paste images side by side with labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            for i, (img, label) in enumerate(zip(images, labels)):
                x_offset = i * width
                combined_img.paste(img, (x_offset, 30))
                
                # Add label
                if font:
                    draw.text((x_offset + 5, 5), label, fill='black', font=font)
                else:
                    draw.text((x_offset + 5, 5), label, fill='black')
            
            # Save the combined image
            debug_dir = os.path.join(output_dir, 'debug_viz')
            os.makedirs(debug_dir, exist_ok=True)
            
            save_path = os.path.join(debug_dir, f'debug_step_{global_step:06d}.jpg')
            combined_img.save(save_path, 'JPEG', quality=95)
            
            print(f"Debug visualization saved: {save_path}")
            
    except Exception as e:
        print(f"Error in debug visualization: {e}")
