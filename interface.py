import gradio as gr
import torch
from PIL import Image
from transformers import CLIPTokenizer

# Import your existing model and pipeline modules
import load_model
import pipeline

# Device Configuration
ALLOW_CUDA = True
ALLOW_MPS = False

def determine_device():
    if torch.cuda.is_available() and ALLOW_CUDA:
        return "cuda"
    elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
        return "mps"
    return "cpu"

DEVICE = determine_device()
print(f"Using device: {DEVICE}")

# Load tokenizer and models
tokenizer = CLIPTokenizer("vocab.json", merges_file="merges.txt")
model_file = "inkpunk-diffusion-v1.ckpt"
models = load_model.preload_models_from_standard_weights(model_file, DEVICE)
# models=None

def generate_image(
    prompt, 
    uncond_prompt="", 
    do_cfg=True, 
    cfg_scale=8, 
    sampler="ddpm", 
    num_inference_steps=50, 
    seed=42, 
    input_image=None, 
    strength=1.0
):
    """
    Generate an image using the Stable Diffusion pipeline
    
    Args:
    - prompt (str): Text description of the image to generate
    - uncond_prompt (str, optional): Negative prompt to guide generation
    - do_cfg (bool): Whether to use classifier-free guidance
    - cfg_scale (float): Classifier-free guidance scale
    - sampler (str): Sampling method
    - num_inference_steps (int): Number of denoising steps
    - seed (int): Random seed for reproducibility
    - input_image (PIL.Image, optional): Input image for image-to-image generation
    - strength (float): Strength of image transformation (0-1)
    
    Returns:
    - PIL.Image: Generated image
    """
    try:
        # Ensure input_image is None if not provided
        if input_image is None:
            strength = 1.0
        
        # Generate the image
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cuda",
            tokenizer=tokenizer,
        )
        
        # Convert numpy array to PIL Image
        return Image.fromarray(output_image)
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def launch_gradio_interface():
    """
    Create and launch Gradio interface for Stable Diffusion
    """
    with gr.Blocks(title="Stable Diffusion Image Generator") as demo:
        gr.Markdown("# 🎨 Stable Diffusion Image Generator")
        
        with gr.Row():
            with gr.Column():
                # Text Inputs
                prompt = gr.Textbox(label="Prompt", 
                                    placeholder="Describe the image you want to generate...")
                uncond_prompt = gr.Textbox(label="Negative Prompt (Optional)", 
                                            placeholder="Describe what you don't want in the image...")
                
                # Generation Parameters
                with gr.Accordion("Advanced Settings", open=False):
                    do_cfg = gr.Checkbox(label="Use Classifier-Free Guidance", value=True)
                    cfg_scale = gr.Slider(minimum=1, maximum=14, value=8, label="CFG Scale")
                    sampler = gr.Dropdown(
                        choices=["ddpm", "ddim", "pndm"],  # Add more samplers if available
                        value="ddpm", 
                        label="Sampling Method"
                    )
                    num_inference_steps = gr.Slider(
                        minimum=10, 
                        maximum=100, 
                        value=50, 
                        label="Number of Inference Steps"
                    )
                    seed = gr.Number(value=42, label="Random Seed")
                
                # Image-to-Image Section
                with gr.Accordion("Image-to-Image", open=False):
                    input_image = gr.Image(type="pil", label="Input Image (Optional)")
                    strength = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        value=0.8, 
                        label="Image Transformation Strength"
                    )
                
                # Generate Button
                generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Row():
            # Output Image
            output_image = gr.Image(label="Generated Image")
        
        # Connect Button to Generation Function
        generate_btn.click(
            fn=generate_image, 
            inputs=[
                prompt, uncond_prompt, do_cfg, cfg_scale, 
                sampler, num_inference_steps, seed, 
                input_image, strength
            ],
            outputs=output_image
        )
    
    # Launch the interface
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio_interface()