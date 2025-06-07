# stable-diffusion

# 🎨 Stable Diffusion & CatVTON Implementation

<div align="center">

![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-From%20Scratch-blue?style=for-the-badge&logo=pytorch)
![CatVTON](https://img.shields.io/badge/CatVTON-Virtual%20Try--On-purple?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10.9-green?style=for-the-badge&logo=python&logoColor=white)

*A comprehensive implementation of Stable Diffusion from scratch with CatVTON virtual try-on capabilities*

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Features](#-features)
- [⚙️ Setup & Installation](#️-setup--installation)
- [📥 Model Downloads](#-model-downloads)
- [🎯 CatVTON Integration](#-catvton-integration)
- [📚 References](#-references)
- [👤 Author](#-author)
- [📜 License](#-license)

---

## 🌟 Overview

This project implements **Stable Diffusion from scratch** using PyTorch, with an additional **CatVTON (Virtual Cloths Try-On)** model built on top of stable-diffusion. The implementation includes:

- ✨ Complete Stable Diffusion pipeline built from ground up **(Branch: Main)**
- 🎭 CatVTON model for virtual garment try-on **(Branch: CatVTON)**
- 🧠 Custom attention mechanisms and CLIP integration
- 🔄 DDPM (Denoising Diffusion Probabilistic Models) implementation
- 🖼️ Inpainting capabilities using pretrained weights

---

## 🏗️ Project Structure

```
stable-diffusion/
├── 📁 Core Components
│   ├── attention.py          # Attention mechanisms
│   ├── clip.py              # CLIP model implementation
│   ├── ddpm.py              # DDPM sampler
│   ├── decoder.py           # VAE decoder
│   ├── encoder.py           # VAE encoder
│   ├── diffusion.py         # Diffusion process
│   ├── model.py             # Defining model & loading pre-trained weights
│   └── pipeline.py          # Main pipeline
│
├── 📁 Utilities & Interface
│   ├── interface.py         # User interface
│   ├── model_converter.py   # Model conversion utilities
│   └── requirements.txt     # Dependencies
│
├── 📁 Data & Models
│   ├── vocab.json           # Tokenizer vocabulary
│   ├── merges.txt           # BPE merges
│   ├── inkpunk-diffusion-v1.ckpt     # Inkpunk model weights
│   └── sd-v1-5-inpainting.ckpt      # Inpainting model weights
│
├── 📁 Sample Data
│   ├── person.jpg           # Person image for try-on
│   ├── garment.jpg          # Garment image
│   ├── agnostic_mask.png    # Segmentation mask
│   ├── dog.jpg              # Test image
│   ├── image.png            # Generated sample
│   └── zalando-hd-resized.zip # Dataset
│
└── 📁 Notebooks & Documentation
    ├── test.ipynb           # Testing notebook
    └── README.md            # This file
```

---

## 🚀 Features

### 🎨 Stable Diffusion Core
- **From-scratch implementation** of Stable Diffusion architecture
- **Custom CLIP** text encoder integration
- **VAE encoder/decoder** for latent space operations
- **DDPM sampling** with configurable steps
- **Attention mechanisms** optimized for diffusion

### 👕 CatVTON Capabilities
- **Virtual garment try-on** using inpainting
- **Person-garment alignment** and fitting
- **Mask-based inpainting** for realistic results

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10.9
- CUDA-compatible GPU (recommended)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Harsh-Kesharwani/stable-diffusion.git
cd stable-diffusion
git checkout CatVTON  # Switch to CatVTON branch to use virtual-try-on model
```

### 2. Create Virtual Environment
```bash
conda create -n stable-diffusion python=3.10.9
conda activate stable-diffusion
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📥 Model Downloads

### Required Files

#### 1. Tokenizer Files
Download from [Stable Diffusion v1.4 Tokenizer](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/tokenizer):
- `vocab.json`
- `merges.txt`

#### 2. Model Checkpoints
- **Inkpunk Diffusion**: Download `inkpunk-diffusion-v1.ckpt` from [Envvi/Inkpunk-Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main)
- **Inpainting Model**: Download from [Stable Diffusion v1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

### Download Script
```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download tokenizer files
wget -O vocab.json "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/vocab.json"
wget -O merges.txt "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/merges.txt"

# Note: Large model files need to be downloaded manually from HuggingFace
```

---

### Interactive Interface
```bash
python interface.py
```

---

## 🎯 CatVTON Integration

The CatVTON model extends the base Stable Diffusion with specialized capabilities for virtual garment try-on:

### Key Components
1. **Inpainting Pipeline**: Uses `sd-v1-5-inpainting.ckpt` for mask-based generation
2. **Garment Alignment**: Automatic alignment of garments to person pose
3. **Mask Generation**: Automated or manual mask creation for try-on regions
---

## 📚 References

### 📖 Implementation Guides
- [Implementing Stable Diffusion from Scratch - Medium](https://medium.com/@sayedebad.777/implementing-stable-diffusion-from-scratch-using-pytorch-f07d50efcd97)
- [Stable Diffusion Implementation - YouTube](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

### 🤗 HuggingFace Resources
- [Diffusers: Adapt a Model](https://huggingface.co/docs/diffusers/training/adapt_a_model)
- [Stable Diffusion v1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- [CompVis Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [Inkpunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion)

### 📄 Academic Papers
- Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models
- DDPM: Denoising Diffusion Probabilistic Models
- CatVTON: Category-aware Virtual Try-On Network

---

## 👤 Author

<div align="center">

**Harsh Kesharwani**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Harsh-Kesharwani)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-kesharwani/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:harshkesharwani777@gmail.com)

*Passionate about AI, Computer Vision, and Generative Models*

</div>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- CompVis team for the original Stable Diffusion implementation
- HuggingFace for providing pre-trained weights, dataset and references.
- The open-source community for various implementations and tutorials
- Zalando Research for the fashion dataset

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ by [Harsh Kesharwani](https://www.linkedin.com/in/harsh-kesharwani/)*

</div>


<!-- 1. Download `vocab.json` and `merges.txt` from https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/tokenizer and save them in the `data` folder
1. Download `inkpunk-diffusion-v1.ckpt` from https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main and save it in the `data` folder -->

<!-- IMPORTANT REFRRENCE
3. https://huggingface.co/docs/diffusers/training/adapt_a_model -->
<!-- 4. https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting -->