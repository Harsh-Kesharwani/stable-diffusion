# 🎨 CatVTON - Virtual Clothes Try-On Implementation

<div align="center">

![CatVTON](https://img.shields.io/badge/CatVTON-Virtual%20Try--On-purple?style=for-the-badge)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-From%20Scratch-blue?style=for-the-badge&logo=pytorch)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white)

*A complete implementation of CatVTON research paper for virtual clothes try-on using Stable Diffusion*

</div>

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Features](#features)
* [Setup & Installation](#setup--installation)
* [Model Downloads](#model-downloads)
* [Usage](#usage)
* [CatVTON Approaches](#catvton-approaches)
* [Training](#training)
* [Dataset](#dataset)
* [References](#references)
* [Author](#author)
* [License](#license)

---

## Overview

This project is a **from-scratch implementation of the CatVTON research paper** for virtual clothes try-on. It combines the power of Stable Diffusion with specialized virtual try-on capabilities, supporting both mask-based and mask-free inference approaches.

**Key Highlights:**
* Complete CatVTON implementation from research paper
* Two inference modes: Mask-based and Mask-free
* Built on Stable Diffusion architecture
* Custom training pipeline for virtual try-on
* Comprehensive dataset handling for VITON

---

## Project Structure

```text
stable-diffusion/
├── Core Models & Components
│   ├── CatVTON_model.py      # Main CatVTON model implementation
│   ├── attention.py          # Multi-head attention mechanisms
│   ├── clip.py               # CLIP text encoder
│   ├── ddpm.py               # DDPM denoising scheduler
│   ├── decoder.py            # VAE decoder
│   ├── encoder.py            # VAE encoder  
│   ├── diffusion.py          # Diffusion process logic
│   ├── load_model.py         # Model checkpoint loading utilities
│   └── utils.py              # Helper functions and utilities
│
├── Training & Data Processing
│   ├── training.py           # Training script
│   ├── training.ipynb        # Interactive training notebook
│   ├── VITON_Dataset.py      # Dataset loader for VITON format
│   ├── sample_dataset/       # Sample training data
│   └── model_converter.py    # Checkpoint conversion utilities
│
├── Inference & Applications
│   ├── app.py                # Main application interface
│   ├── mask_based_inference.ipynb    # Mask-based try-on inference
│   ├── mask_free_inference.ipynb     # Mask-free try-on inference
│   ├── mask-based-output/    # Generated results (mask-based)
│   └── mask-free-output/     # Generated results (mask-free)
│
├── Configuration Files
│   ├── vocab.json            # CLIP tokenizer vocabulary
│   ├── merges.txt            # CLIP tokenizer merges
│   ├── requirements.txt      # Python dependencies
│   └── logs.txt              # Training and execution logs
│
└── Documentation
    └── README.md             # This file
```

---

## Features

### Virtual Try-On Capabilities
* **Mask-Based Approach**: Precise garment replacement using segmentation masks
* **Mask-Free Approach**: Direct garment swapping without explicit masking
* **Pose-Aware Fitting**: Maintains human pose and body structure
* **High-Quality Output**: Realistic garment integration and lighting

### Technical Implementation
* Complete CatVTON architecture from research paper
* Custom Stable Diffusion pipeline built from scratch
* Advanced attention mechanisms for garment-person alignment
* Flexible training pipeline with custom dataset support
* Efficient inference with optimized memory usage

---

## Setup & Installation

### Prerequisites

* **Python**: 3.10 or higher
* **GPU**: CUDA-compatible GPU with 8GB+ VRAM (recommended)
* **Storage**: 15GB+ free space for models and datasets
* **Memory**: 16GB+ RAM recommended

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/Harsh-Kesharwani/stable-diffusion.git
cd stable-diffusion
git checkout CatVTON  # Switch to CatVTON branch
```

2. **Create Virtual Environment**
```bash
# Using conda (recommended)
conda create -n catvton python=3.10
conda activate catvton

# Or using venv
python -m venv catvton
source catvton/bin/activate  # Linux/Mac
# catvton\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Model Downloads

### Automatic Download (Recommended)

The models will be automatically downloaded when you run the inference notebooks. However, you can also download them manually:

### Manual Download

#### For Mask-Based Approach:
```bash
# Base inpainting model
wget -O sd-v1-5-inpainting.ckpt "https://huggingface.co/sd-legacy/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"

# Fine-tuned CatVTON weights
wget -O finetuned_weights.safetensors "https://huggingface.co/harsh99/virtual-cloths-try-on/resolve/main/finetuned_weights.safetensors"
```

#### For Mask-Free Approach:
```bash
# Base instruct-pix2pix model
wget -O instruct-pix2pix-00-22000.ckpt "https://huggingface.co/timbrooks/instruct-pix2pix/resolve/main/instruct-pix2pix-00-22000.ckpt"

# Fine-tuned mask-free weights
wget -O maskfree_finetuned_weights.safetensors "https://huggingface.co/harsh99/virtual-cloths-try-on/resolve/main/maskfree_finetuned_weights.safetensors"
```

#### Tokenizer Files:
```bash
# CLIP tokenizer files (required for both approaches)
wget -O vocab.json "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/vocab.json"
wget -O merges.txt "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/merges.txt"
```

### Model Storage Structure
```text
stable-diffusion/
├── sd-v1-5-inpainting.ckpt              # Mask-based base model
├── finetuned_weights.safetensors         # Mask-based fine-tuned weights
├── instruct-pix2pix-00-22000.ckpt      # Mask-free base model
├── maskfree_finetuned_weights.safetensors # Mask-free fine-tuned weights
├── vocab.json                            # Tokenizer vocabulary
└── merges.txt                           # Tokenizer merges
```

---

## Usage

### Quick Start with Jupyter Notebooks

#### 1. Mask-Based Virtual Try-On
```bash
jupyter notebook mask_based_inference.ipynb
```
* Uses segmentation masks for precise garment placement
* Better for complex garments and detailed fitting
* Requires person and garment images + optional mask

#### 2. Mask-Free Virtual Try-On  
```bash
jupyter notebook mask_free_inference.ipynb
```
* Direct garment replacement without masks
* Faster inference and simpler setup
* Only requires person and garment images

### Running the Main Application
```bash
python app.py
```

---

## Training

### Training Dataset
Download HD-VITON dataset from Kaggle.

### Start Training

```bash
# Using training script
python training.py 

# Or using interactive notebook
jupyter notebook training.ipynb
```

### Training Configuration
Key parameters in training:
* `batch_size`: Adjust based on GPU memory (default: 4)
* `learning_rate`: Start with 1e-5 for fine-tuning
* `num_epochs`: 50-100 epochs typically sufficient

---

## Dataset

### VITON Dataset Format
This implementation supports the standard VITON dataset format:

```text
dataset/
├── train/
│   ├── image/              # Person images (256x192 or 512x384)
│   ├── cloth/              # Garment images  
│   ├── image-parse/        # Person parsing masks
│   ├── cloth-mask/         # Garment masks
│   └── train_pairs.txt     # Image-cloth pairs
├── test/
│   └── ...                 # Same structure
├── train_pair.txt
└── test_pair.txt
```

### Custom Dataset
To use your own dataset, ensure:
1. Images are properly sized and aligned
2. Follow the naming convention in `pairs.txt`
3. Include segmentation masks for mask-based approach
4. Update `VITON_Dataset.py` if needed

---

## Performance & Requirements

### System Requirements
* **Minimum**: 8GB GPU VRAM, 16GB RAM
* **Recommended**: 12GB+ GPU VRAM, 32GB RAM
* **Storage**: 15GB for models, additional for datasets

---

## References

### Research Papers
* **CatVTON**: *Concatenation-based Virtual Try-On Network* - Original research paper
* **Stable Diffusion**: *High-Resolution Image Synthesis with Latent Diffusion Models*
* **DDPM**: *Denoising Diffusion Probabilistic Models*

### Model Sources
* [Stable Diffusion Inpainting](https://huggingface.co/sd-legacy/stable-diffusion-inpainting)
* [Instruct-Pix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix)
* [Fine-tuned Weights](https://huggingface.co/harsh99/virtual-cloths-try-on)

### Educational Resources
* [Stable Diffusion from Scratch (Medium)](https://medium.com/@sayedebad.777/implementing-stable-diffusion-from-scratch-using-pytorch-f07d50efcd97)
* [Diffusion Models Explained](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Author

<div align="center">

**Harsh Kesharwani**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Harsh-Kesharwani)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-kesharwani/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:harshkesharwani777@gmail.com)

*AI Researcher & Computer Vision Enthusiast*  
*Specializing in Generative Models and Virtual Try-On Technology*

</div>

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Citation
If you use this code in your research, please cite:
```bibtex
@misc{kesharwani2024catvton,
  title={CatVTON: Virtual Clothes Try-On Implementation},
  author={Harsh Kesharwani},
  year={2024},
  url={https://github.com/Harsh-Kesharwani/stable-diffusion}
}
```

---

## Acknowledgments

* Original CatVTON research team for the groundbreaking paper
* CompVis team for Stable Diffusion architecture
* HuggingFace for model hosting and APIs
* RunwayML for inpainting model contributions
* Open-source AI community for continuous innovation

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ for the AI community*

[🚀 **Try CatVTON Now**](#usage) | [📚 **Read the Docs**](#table-of-contents) | [🤝 **Contribute**](#contributing)

</div>