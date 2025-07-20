import torch
import os
import random
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np
from PIL import Image
from tqdm import tqdm
from VITON_Dataset import VITONHDTestDataset

# Import your custom modules
from load_model import preload_models_from_standard_weights
from ddpm import DDPMSampler
from utils import check_inputs, get_time_embedding, prepare_image, prepare_mask_image, save_debug_visualization, compute_vae_encodings
from diffusers.utils.torch_utils import randn_tensor


class CatVTONTrainer:
    """Simplified CatVTON Training Class with PEFT, CFG and DREAM support"""
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-5,
        num_epochs: int = 50,
        save_steps: int = 1000,
        output_dir: str = "./checkpoints",
        cfg_dropout_prob: float = 0.1,
        max_grad_norm: float = 1.0,
        use_peft: bool = True,
        dream_lambda: float = 10.0,
        resume_from_checkpoint: Optional[str] = None,
        use_mixed_precision: bool = True,
        height=512,
        width=384,
    ):
        self.training = True
        self.models = models
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.cfg_dropout_prob = cfg_dropout_prob
        self.max_grad_norm = max_grad_norm
        self.use_peft = use_peft
        self.dream_lambda = dream_lambda
        self.use_mixed_precision = use_mixed_precision
        self.height = height
        self.width = width
        self.generator = torch.Generator(device=device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup mixed precision training
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.weight_dtype = torch.float16 if use_mixed_precision else torch.float32
        
        # Initialize scheduler and sampler
        self.scheduler = DDPMSampler(self.generator, num_training_steps=1000)

        # Resume from checkpoint if provided
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup models and optimizers
        self._setup_training()
        
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            
        
    
        self.encoder = self.models.get('encoder', None)
        self.decoder = self.models.get('decoder', None)
        self.diffusion = self.models.get('diffusion', None)
    
    def _setup_training(self):
        """Setup models for training with PEFT"""
        # Move models to device
        for name, model in self.models.items():
            model.to(self.device)
        
        # Freeze all parameters first
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
        
        # Enable training for specific layers based on PEFT strategy
        if self.use_peft:
            self._enable_peft_training()
        else:
            # Enable full training for diffusion model
            for param in self.diffusion.parameters():
                param.requires_grad = True
        
        # Collect trainable parameters
        trainable_params = []
        total_params = 0
        trainable_count = 0
        
        for name, model in self.models.items():
            for param_name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params.append(param)
                    trainable_count += param.numel()

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
        
        # Setup optimizer - AdamW as per paper
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        # Setup learning rate scheduler (constant)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1.0
        )
    
    def _enable_peft_training(self):
        """Enable PEFT training - only self-attention layers"""
        print("Enabling PEFT training (self-attention layers only)")
        
        unet = self.models['diffusion'].unet
        
        # Enable attention layers in encoders and decoders
        for layers in [unet.encoders, unet.decoders]:
            for layer in layers:
                for module_idx, module in enumerate(layer):
                    for name, param in module.named_parameters():
                        if 'attention_1' in name:
                            param.requires_grad = True
                        
        # Enable attention layers in bottleneck
        for layer in unet.bottleneck:
            for name, param in layer.named_parameters():
                if 'attention_1' in name:
                    param.requires_grad = True
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute MSE loss for denoising with DREAM strategy"""
        person_images = batch['person'].to(self.device)
        cloth_images = batch['cloth'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        batch_size = person_images.shape[0]

        concat_dim = -2  # y axis concat
        
        # Prepare inputs
        image, condition_image, mask = check_inputs(person_images, cloth_images, masks, self.width, self.height)
        image = prepare_image(person_images).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(cloth_images).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(masks).to(self.device, dtype=self.weight_dtype)
        
        # Mask image
        masked_image = image * (mask < 0.5)

        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            # VAE encoding
            masked_latent = compute_vae_encodings(masked_image, self.encoder)
            person_latent = compute_vae_encodings(person_images, self.encoder)
            condition_latent = compute_vae_encodings(condition_image, self.encoder)
            mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
            
            
            del image, mask, condition_image
            
            # Apply CFG dropout to garment latent (10% chance)
            if self.training and random.random() < self.cfg_dropout_prob:
                condition_latent = torch.zeros_like(condition_latent)
            
            # Concatenate latents
            input_latents = torch.cat([masked_latent, condition_latent], dim=concat_dim)
            mask_input = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
            target_latents = torch.cat([person_latent, condition_latent], dim=concat_dim)

            noise = randn_tensor(
                target_latents.shape,
                generator=self.generator,
                device=target_latents.device,
                dtype=self.weight_dtype,
            )

            # timesteps = torch.randint(1, 1000, size=(1,), device=self.device)[0].long().item()
            # timesteps = torch.tensor(timesteps, device=self.device)
            # timesteps_embedding = get_time_embedding(timesteps).to(self.device, dtype=self.weight_dtype)
            timesteps = torch.randint(1, 1000, size=(batch_size,))
            timesteps_embedding = get_time_embedding(timesteps).to(self.device, dtype=self.weight_dtype)

            # Add noise to latents
            noisy_latents = self.scheduler.add_noise(target_latents, timesteps, noise)

            # UNet(zt ⊙ Mi ⊙ Xi) where ⊙ is channel concatenation
            unet_input = torch.cat([
                input_latents,      # Xi
                mask_input,         # Mi
                noisy_latents,      # zt
            ], dim=1).to(self.device, dtype=self.weight_dtype)  # Channel dimension
            

            # DREAM strategy implementation
            if self.dream_lambda > 0:
                # Get initial noise prediction
                with torch.no_grad():
                    epsilon_theta = self.diffusion(
                        unet_input,
                        timesteps_embedding
                    )
                
                # DREAM noise combination: ε + λ*εθ
                dream_noise = noise + self.dream_lambda * epsilon_theta
                
                # Create new noisy latents with DREAM noise
                dream_noisy_latents = self.scheduler.add_noise(target_latents, timesteps, dream_noise)

                dream_unet_input = torch.cat([
                    input_latents, 
                    mask_input,
                    dream_noisy_latents
                ], dim=1).to(self.device, dtype=self.weight_dtype)

                predicted_noise = self.diffusion(
                    dream_unet_input,
                    timesteps_embedding
                )
                # DREAM loss: |(ε + λεθ) - εθ(ẑt, t)|²
                loss = F.mse_loss(predicted_noise, dream_noise)
            else:
                # Standard training without DREAM
                predicted_noise = self.diffusion(
                    unet_input,
                    timesteps_embedding,
                )
                
                # Standard MSE loss
                loss = F.mse_loss(predicted_noise, noise)
            
            if self.global_step % 1000 == 0:
                save_debug_visualization(
                    person_images=person_images,
                    cloth_images=cloth_images, 
                    masks=masks,
                    masked_image=masked_image,
                    noisy_latents=noisy_latents,
                    predicted_noise=predicted_noise,
                    target_latents=target_latents,
                    decoder=self.decoder,
                    global_step=self.global_step,
                    output_dir=self.output_dir,
                    device=self.device
                )
        return loss
    
    def train_epoch(self) -> float:
        """Train for one epoch - simplified version"""
        self.diffusion.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for step, batch in enumerate(self.train_dataloader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(batch)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping and optimizer step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.diffusion.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.compute_loss(batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.diffusion.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate
            self.lr_scheduler.step()
            self.global_step += 1
            
            total_loss += loss.item()
            
            # Update progress bar
            # progress_bar.set_postfix({
            #     'loss': loss.item(),
            #     'lr': self.optimizer.param_groups[0]['lr'],
            #     'step': self.global_step
            # })
            
            # Save checkpoint based on steps
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint()
            
            # Clear cache periodically to prevent OOM
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop - simplified version"""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total training batches per epoch: {len(self.train_dataloader)}")
        print(f"Using DREAM with lambda = {self.dream_lambda}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}")
            
            # Save epoch checkpoint
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self._save_checkpoint(epoch_checkpoint=True)
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        print("Training completed!")
    
    def _save_checkpoint(self, is_best: bool = False, epoch_checkpoint: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'diffusion_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'dream_lambda': self.dream_lambda,
            'use_peft': self.use_peft,
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_final:
            checkpoint_path = self.output_dir / "final_model.pth"
        elif is_best:
            checkpoint_path = self.output_dir / "best_model.pth"
        elif epoch_checkpoint:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch+1}.pth"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.dream_lambda = checkpoint.get('dream_lambda', 10.0)
        
        self.models['diffusion'].load_state_dict(checkpoint['diffusion_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")


def create_dataloaders(args) -> DataLoader:
    """Create training dataloader"""
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}.")
    
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    return dataloader

def main():
    args = argparse.Namespace()
    args.__dict__ = {
        "base_model_path": "sd-v1-5-inpainting.ckpt",
        "dataset_name": "vitonhd",
        "data_root_path": "./viton-hd-dataset",
        "output_dir": "./checkpoints",
        "resume_from_checkpoint": "./checkpoints/checkpoint_step_50000.pth",
        "seed": 42,
        "batch_size": 2,
        "width": 384,
        "height": 384,
        "repaint": True,
        "eval_pair": True,
        "concat_eval_results": True,
        "concat_axis": 'y',
        "device": "cuda",
        "num_epochs": 50,  
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
        "cfg_dropout_prob": 0.1,
        "dream_lambda": 10.0,
        "use_peft": True,
        "use_mixed_precision": True,
        "save_steps": 10000,
        "is_train": True
    }
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    torch.set_float32_matmul_precision("high")
    
    print("-"*100)

    # Load pretrained models
    print("Loading pretrained models...")
    models = preload_models_from_standard_weights(args.base_model_path, args.device)
    print("Models loaded successfully.")
    
    print("-"*100)
    
    # Create dataloader
    print("Creating dataloader...")
    train_dataloader = create_dataloaders(args)
    
    print(f"Training for {args.num_epochs} epochs")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    print("-"*100)
    
    # Initialize trainer
    print("Initializing trainer...")    
    trainer = CatVTONTrainer(
        models=models,
        train_dataloader=train_dataloader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        cfg_dropout_prob=args.cfg_dropout_prob,
        max_grad_norm=args.max_grad_norm,
        use_peft=args.use_peft,
        dream_lambda=args.dream_lambda,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_mixed_precision=args.use_mixed_precision,
        height=args.height,
        width=args.width
    )
    
    # Start training
    print("Starting training...")
    trainer.train() 

if __name__ == "__main__":
    main()