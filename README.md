# 🎨 Stable Diffusion from Scratch

<div align="center">

![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-From%20Scratch-blue?style=for-the-badge&logo=pytorch)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white)

*A complete implementation of Stable Diffusion built from scratch using PyTorch*

</div>

## Overview

This repository contains a **from-scratch implementation** of Stable Diffusion using PyTorch. All core components including VAE, CLIP, U-Net, and DDPM sampling are implemented without relying on external diffusion libraries.

## Project Structure

```
stable-diffusion/
├── attention.py          # Self-attention and cross-attention mechanisms
├── clip.py               # CLIP text encoder implementation
├── ddpm.py               # DDPM sampling scheduler
├── decoder.py            # VAE decoder
├── encoder.py            # VAE encoder  
├── diffusion.py          # U-Net diffusion model
├── model.py              # Model loading utilities
├── pipeline.py           # Main diffusion pipeline
├── interface.py          # Interactive generation script
├── test.ipynb            # Jupyter notebook for testing
├── vocab.json            # CLIP tokenizer vocabulary
├── merges.txt            # CLIP tokenizer merges
└── requirements.txt      # Python dependencies
```

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/Harsh-Kesharwani/stable-diffusion.git
cd stable-diffusion
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
```bash
# Download Inkpunk Diffusion model
wget -O inkpunk-diffusion-v1.ckpt "https://huggingface.co/Envvi/Inkpunk-Diffusion/resolve/main/inkpunk-diffusion-v1.ckpt?download=true"
```

### 4. Run Inference
```bash
python interface.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook test.ipynb
```

## Features

- ✅ **Complete from-scratch implementation**
- ✅ **VAE Encoder/Decoder** for latent space conversion
- ✅ **CLIP Text Encoder** for text conditioning
- ✅ **U-Net Diffusion Model** with attention mechanisms
- ✅ **DDPM Sampling** scheduler
- ✅ **Text-to-image generation**
- ✅ **Custom attention layers** (self-attention & cross-attention)

## Usage

The `interface.py` script provides an interactive way to generate images:

```python
# Example prompt
prompt = "a beautiful landscape, digital art, trending on artstation"
```

## Model Details

- **Base Model**: Inkpunk Diffusion v1 (fine-tuned Stable Diffusion)
- **Resolution**: 512x512
- **Scheduler**: DDPM with 1000 timesteps
- **Text Encoder**: CLIP ViT-L/14

## Requirements

- Python 3.10+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory

## Author

**Harsh Kesharwani**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Harsh-Kesharwani)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-kesharwani/)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ for learning and understanding diffusion models*

</div>