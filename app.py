import os
import torch
import gradio as gr
from PIL import Image
import numpy as np
from typing import Optional

# Import your custom modules
from load_model import preload_models_from_standard_weights
from utils import to_pil_image

import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from diffusers.utils.torch_utils import randn_tensor

from utils import (check_inputs_maskfree, get_time_embedding, numpy_to_pil, prepare_image, compute_vae_encodings)
from ddpm import DDPMSampler


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
        image_latent = compute_vae_encodings(image, self.encoder)
        condition_latent = compute_vae_encodings(condition_image, self.encoder)
        
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
        image = self.decoder(latents.to(self.device, dtype=self.weight_dtype))
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        return image
    
def load_models():
    try:
        print("🚀 Starting model loading process...")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"Available CUDA memory: {free_memory / 1e9:.2f} GB")
        
        device = "cuda" if cuda_available else "cpu"
        
        # Check if model files exist
        ckpt_path = "instruct-pix2pix-00-22000.ckpt"
        finetune_path = "maskfree_finetuned_weights.safetensors"
        
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint file not found: {ckpt_path}")
            return None, None
            
        if not os.path.exists(finetune_path):
            print(f"❌ Finetune weights file not found: {finetune_path}")
            return None, None
        
        print("📦 Loading models from weights...")
        
        models = preload_models_from_standard_weights(
            ckpt_path=ckpt_path, 
            device=device, 
            finetune_weights_path=finetune_path
        )
        
        if not models:
            print("❌ Failed to load models")
            return None, None
        
        # Convert all models to consistent dtype to avoid mixed precision issues
        weight_dtype = torch.float32  # Use float32 to avoid dtype mismatch
        print(f"Converting models to {weight_dtype}...")
        
        # Ensure all models use the same dtype
        for model_name, model in models.items():
            if model is not None:
                try:
                    model = model.to(dtype=weight_dtype)
                    models[model_name] = model
                    print(f"✅ {model_name} converted to {weight_dtype}")
                except Exception as e:
                    print(f"⚠️ Could not convert {model_name} to {weight_dtype}: {e}")
        
        print("🔧 Initializing pipeline...")
        
        pipeline = CatVTONPix2PixPipeline(
            weight_dtype=weight_dtype,
            device=device,
            skip_safety_check=True,
            models=models,
        )
        
        print("✅ Models and pipeline loaded successfully!")
        return models, pipeline
        
    except Exception as e:
        print(f"❌ Error in load_models: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
def person_example_fn(image_path):
    """Handle person image examples"""
    if image_path:
        return image_path
    return None

def create_demo(pipeline=None):
    """Create the Gradio interface"""
    
    def submit_function_p2p(
        person_image_path: Optional[str],
        cloth_image_path: Optional[str], 
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = 42,
    ) -> Optional[Image.Image]:
        """Process virtual try-on inference"""
        
        try:
            if not person_image_path or not cloth_image_path:
                gr.Warning("Please upload both person and cloth images!")
                return None
            
            if not os.path.exists(person_image_path):
                gr.Error("Person image file not found!")
                return None
                
            if not os.path.exists(cloth_image_path):
                gr.Error("Cloth image file not found!")
                return None
            
            if pipeline is None:
                gr.Error("Models not loaded! Please restart the application.")
                return None
            
            # Load images
            try:
                person_image = Image.open(person_image_path).convert('RGB')
                cloth_image = Image.open(cloth_image_path).convert('RGB')
            except Exception as e:
                gr.Error(f"Error loading images: {str(e)}")
                return None
            
            # Set up generator
            generator = torch.Generator(device=pipeline.device)
            if seed != -1:
                generator.manual_seed(seed)
            
            print("🔄 Processing virtual try-on...")
            
            # Run inference
            with torch.no_grad():
                results = pipeline(
                    person_image,
                    cloth_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512,
                    width=384,
                    generator=generator,
                )
            
            # Process results
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
            else:
                result = results
                
            return result
                
        except Exception as e:
            print(f"❌ Error in submit_function_p2p: {e}")
            import traceback
            traceback.print_exc()
            gr.Error(f"Error during inference: {str(e)}")
            return None
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .image-container {
        max-height: 600px;
    }
    """
    
    with gr.Blocks(css=css, title="Virtual Try-On") as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🧥 Virtual Try-On with CatVTON</h1>
            <p>Upload a person image and a clothing item to see how they look together!</p>
        </div>
        """)
        
        with gr.Tab("Mask-Free Virtual Try-On"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    with gr.Row():
                        image_path_p2p = gr.Image(
                            type="filepath",
                            interactive=True,
                            visible=False,
                        )
                        person_image_p2p = gr.Image(
                            interactive=True, 
                            label="Person Image", 
                            type="filepath",
                            elem_classes=["image-container"]
                        )
                    
                    with gr.Row():
                        cloth_image_p2p = gr.Image(
                            interactive=True, 
                            label="Clothing Image", 
                            type="filepath",
                            elem_classes=["image-container"]
                        )
                    
                    submit_p2p = gr.Button("✨ Generate Try-On", variant="primary", size="lg")
                    
                    gr.Markdown(
                        '<center><span style="color: #FF6B6B; font-weight: bold;">⚠️ Click only once and wait for processing!</span></center>'
                    )
                    
                    with gr.Accordion("🔧 Advanced Options", open=False):
                        num_inference_steps_p2p = gr.Slider(
                            label="Inference Steps", 
                            minimum=10, 
                            maximum=100, 
                            step=5, 
                            value=50,
                            info="More steps = better quality but slower"
                        )
                        guidance_scale_p2p = gr.Slider(
                            label="Guidance Scale", 
                            minimum=0.0, 
                            maximum=7.5, 
                            step=0.5, 
                            value=2.5,
                            info="Higher values = stronger conditioning"
                        )
                        seed_p2p = gr.Slider(
                            label="Seed", 
                            minimum=-1, 
                            maximum=10000, 
                            step=1, 
                            value=42,
                            info="Use -1 for random seed"
                        )
                
                with gr.Column(scale=2, min_width=500):
                    result_image_p2p = gr.Image(
                        interactive=False, 
                        label="Result (Person | Clothing | Generated)",
                        elem_classes=["image-container"]
                    )
                    
                    gr.Markdown("""
                    ### 📋 Instructions:
                    1. Upload a **person image** (front-facing works best)
                    2. Upload a **clothing item** you want to try on
                    3. Adjust advanced settings if needed
                    4. Click "Generate Try-On" and wait
                    
                    ### 💡 Tips:
                    - Use clear, high-resolution images
                    - Person should be facing forward
                    - Clothing items work best when laid flat or on a model
                    - Try different seeds if you're not satisfied with results
                    """)
        
        # Event handlers
        image_path_p2p.change(
            person_example_fn, 
            inputs=image_path_p2p, 
            outputs=person_image_p2p
        )
        
        submit_p2p.click(
            submit_function_p2p,
            inputs=[
                person_image_p2p,
                cloth_image_p2p,
                num_inference_steps_p2p,
                guidance_scale_p2p,
                seed_p2p,
            ],
            outputs=result_image_p2p,
        )
    
    return demo

def app_gradio():
    """Main application function"""
    
    # Load models at startup
    print("🚀 Loading models...")
    models, pipeline = load_models()
    if not models or not pipeline:
        print("❌ Failed to load models. Please check your model files.")
        return
    
    # Create and launch demo
    demo = create_demo(pipeline=pipeline)
    demo.launch(
        share=True, 
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    app_gradio()