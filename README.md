# VALID: A self-supervised Volumetric biomedicAL Imaging Denoiser
<p align="center">
    <img src="./resource/logo_lr.png" alt="VALID Logo" width="600"/>
</p>
![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Advancing Biomedical Optical Volumetric Image Denoising via Self-Supervised Orthogonal Learning (https://doi.org/10.21203/rs.3.rs-6101322/v1)

## ✨ Key Features

- 🚀 **Zero-shot Volumetric denoising**
- 🤖 **Structured-friendly Orthogonal Learningg**
- 🔄 **High Adaptability and Generalization**
- 📊 **GUI is available for both training and testing pipelines**

## 🛠 Installation

### Requirements
- Python 3.9+
- PyTorch 2.x (CUDA version aligned with your hardware)
- CUDA-capable GPU (recommended)

### Quick Setup
```bash
# Create and activate environment
conda create -n VALID python == 3.9.0
conda activate VALID
# configrate pytorch-GPU with CUDA (CUDA version aligned with your hardware) from [here](https://pytorch.org/get-started/locally/).
# Install packages
pip install -r requirements.txt
```

## 🚀 Quick Start

### Command Line Mode to perform VALID GUI
```bash
python run.py
```

### GUI Mode
```bash
Click Training and Testing buttons to perform selected pipeline.
```

## 📁 Directory Structure
```
FAST/
├── checkpoint/         # Model checkpoints
│   └── model_name
├── data/              # Data directory
│   ├── test/          # Testing data
│   └── train/         # Training data
├── datasets/          # Dataset processing
│   ├── data_patchify.py     # Data patchify
│   ├── data_patchify_indices.py     # Data patchify indices
│   ├── data_process.py # Data processing
│   ├── dataset_fs.py     # Data classes using file system (for small single data)
│   ├── dataset_h5.py     # Data classes using h5 file (for large single data, notably, don't stop the progress while the h5 file is writing.)
│   └── sampling.py     # Tetris sampling and low-frequency 3D Hessian calculating
├── models/           # Model architectures
│   ├── loss/         # Some ready-made implementations
│   │   └── loss.py
│   ├── network.py    # implementation of HessianConstraintLoss3D and CRN init.
│   └── model_CRN.py  # baselayers
├── resource/    # logo, icon and more
├── jsons/    # Automatic parameter backup
├── result/    # Default result saving path
├── requirements.txt    # Environment configuration
├── run.py          # The entry point for launching the GUI
├── test_pipeline.py      # GUI and test pipeline
├── train_pipeline.py # GUI and train pipeline
└── utils/           # Utility functions
    ├── config.py    # Configuration utils
    ├── fileSplit.py
    ├── general.py   # General utilities
    └── __init__.py
```

## ⚙️ Configuration

Customize parameters by modifying `params.json`:

```python
{
    "epochs": 100,
    "train_frame_num": 99999, # The max frame number for training.
    "w_patch": 256,
    "h_patch": 256,
    "z_patch": 64, # If the GPU memory is insufficient, it is recommended to reduce this parameter.
    "w_overlap": 0.1,
    "h_overlap": 0.1,
    "z_overlap": 0.1,
    "patch_num": -1, # set this as -1, using all the patches.
    "gpu_ids": [0], # set [0,1,2,3] to use DP with multi-GPU.
    "save_freq": 10, 
    "train_folder": r"./data/train",
    "num_workers": 0, # set 0 for windows.
    "weight_reg": 0.0001, # weight of HessianConstraintLoss3D; it can be adjusted to a larger one to accommodate complex noise. 
}
```
Default parameters: # Unless there are special requirements, it is not recommended to make any modifications.
```python
{
    "data_extension": "tif",
    "withGT": False,
    "lr": 0.0001,
    "amsgrad": True,
    "base_features": 16,
    "n_groups": 4,
    "train": True, # or False
    "test": False, # or True
    "data_type": "3D",
    "seed": 3407,
    "clip_gradients": 20.0,
    "mode": "train", # or "test"
    "batch_size": 1
}
```
## 🤝 Contributing

We welcome contributions, particularly:

1. 🐛 Bug reports and fixes
2. ✨ New feature proposals and implementations
3. 📚 Documentation improvements
4. 🎨 Code optimizations

### Coding Standards
- Use `UpperCamelCase` for class names
- Use `lowercase_with_underscores` for functions and variables
- Include docstrings for core functions
- Follow PEP8 standards (validate using `flake8`)

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. This means you are free to:

- ✅ Use
- ✅ Modify
- ✅ Distribute

But you must:
- ⚠️ Disclose source
- ⚠️ Include original copyright
- ⚠️ Use the same license

See [LICENSE](LICENSE) file for full text.

## ❓ FAQ

<details>
<summary>Coming soon</summary>

</details>



## 📮 Contact

- 📧 Email: guyj23@m.fudan.edu.cn
- 📚 Project Page: [GitHub Repository](https://github.com/FDU-donglab/VALID)
- 🌐 Personal homepage: [It's me](https://guyuanjie.com)
---

### Citation

If you use VALID in your research, please cite our paper:

```bibtex
coming soon
```
