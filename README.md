# 🎨 Stable Diffusion & CatVTON Implementation

<div align="center">

![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-From%20Scratch-blue?style=for-the-badge\&logo=pytorch) <br>
![CatVTON](https://img.shields.io/badge/CatVTON-Virtual%20Try--On-purple?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10.9-green?style=for-the-badge\&logo=python\&logoColor=white)

*A comprehensive implementation of Stable Diffusion from scratch with CatVTON virtual try-on capabilities*

</div>

---

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Features](#features)
* [Setup & Installation](#setup--installation)
* [Model Downloads](#model-downloads)
* [CatVTON Integration](#catvton-integration)
* [References](#references)
* [Author](#author)
* [License](#license)

---

## Overview

This project implements **Stable Diffusion from scratch** using PyTorch, extended with **CatVTON (Virtual Cloth Try-On)** for realistic fashion try-on.

* Complete Stable Diffusion pipeline (Branch: `main`)
* CatVTON virtual try-on extension (Branch: `CatVTON`)
* DDPM-based denoising, VAE, and custom attention
* Inpainting and text-to-image capabilities

---

## Project Structure

```text
stable-diffusion/
├── Core Components
│   ├── attention.py          # Attention mechanisms
│   ├── clip.py               # CLIP model
│   ├── ddpm.py               # DDPM sampler
│   ├── decoder.py            # VAE decoder
│   ├── encoder.py            # VAE encoder
│   ├── diffusion.py          # Diffusion logic
│   ├── model.py              # Weight loading
│   └── pipeline.py           # Main pipeline logic
│
├── Utilities & Interface
│   ├── interface.py          # Interactive script
│   ├── model_converter.py    # Weight conversion utilities
│   └── requirements.txt      # Python dependencies
│
├── Data & Models
│   ├── vocab.json
│   ├── merges.txt
│   ├── inkpunk-diffusion-v1.ckpt
│   └── sd-v1-5-inpainting.ckpt
│
├── Sample Data
│   ├── person.jpg
│   ├── garment.jpg
│   ├── agnostic_mask.png
│   ├── dog.jpg
│   ├── image.png
│   └── zalando-hd-resized.zip
│
└── Notebooks & Docs
    ├── test.ipynb
    └── README.md
```

---

## Features

### Stable Diffusion Core

* From-scratch implementation with modular architecture
* Custom CLIP encoder integration
* Latent space generation using VAE
* DDPM sampling process
* Self-attention mechanisms for denoising

### CatVTON Capabilities

* Virtual try-on using inpainting
* Pose-aligned garment fitting
* Segmentation mask based garment overlay

---

## Setup & Installation

### Prerequisites

* Python 3.10.9
* CUDA-compatible GPU
* Git, Conda or venv

### Clone Repository

```bash
git clone https://github.com/Harsh-Kesharwani/stable-diffusion.git
cd stable-diffusion
git checkout CatVTON  # for try-on features
```

### Create Environment

```bash
conda create -n stable-diffusion python=3.10.9
conda activate stable-diffusion
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Test Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Model Downloads

### Tokenizer Files (from SD v1.4)

* `vocab.json`
* `merges.txt`

Download from: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/tokenizer)

### Model Checkpoints

* `inkpunk-diffusion-v1.ckpt`: [Inkpunk Model](https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main)
* `sd-v1-5-inpainting.ckpt`: [Inpainting Weights](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

### Download Script

```bash
mkdir -p data
wget -O data/vocab.json "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/vocab.json"
wget -O data/merges.txt "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/tokenizer/merges.txt"
```

---

## CatVTON Integration

The CatVTON extension allows realistic cloth try-on using Stable Diffusion inpainting.

### Highlights

* `sd-v1-5-inpainting.ckpt` for image completion
* Garment alignment to human pose
* Agnostic segmentation mask usage

Run the interface:

```bash
python interface.py
```

---

## References

### Articles & Guides

* [Stable Diffusion from Scratch (Medium)](https://medium.com/@sayedebad.777/implementing-stable-diffusion-from-scratch-using-pytorch-f07d50efcd97)
* [YouTube: Diffusion Implementation](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

### HuggingFace Resources

* [Stable Diffusion v1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
* [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
* [Inkpunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion)

### Papers

* Stable Diffusion: Latent Diffusion Models
* DDPM: Denoising Diffusion Probabilistic Models
* CatVTON: Category-aware Try-On Network

---

## Author

<div align="center">

**Harsh Kesharwani**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/Harsh-Kesharwani)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/harsh-kesharwani/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge\&logo=gmail\&logoColor=white)](mailto:harshkesharwani777@gmail.com)

*Passionate about AI, Computer Vision, and Generative Models*

</div>

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* CompVis team for Stable Diffusion
* HuggingFace for models and APIs
* Zalando Research for dataset
* Open-source contributors and educators

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ by [Harsh Kesharwani](https://www.linkedin.com/in/harsh-kesharwani/)*

</div>
