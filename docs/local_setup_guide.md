# DeMemSeg: Local Environment Setup Guide (Without Docker)

This guide provides step-by-step instructions for setting up the DeMemSeg environment directly on your local machine (e.g., Ubuntu or WSL2) without using Docker. 

OpenMMLab libraries (like MMDetection and MMCV) have strict version dependencies regarding CUDA and PyTorch. If you choose to install them manually, please follow these steps carefully to avoid common build and compatibility errors.

## 0. Prerequisites

- **OS**: Linux (Ubuntu 20.04/22.04/24.04, etc.) or WSL2 on Windows.
- **GPU**: NVIDIA GPU.
- **CUDA Toolkit**: Ensure you have a CUDA version installed that matches the version PyTorch expects (e.g., CUDA 12.8).

### ⚠️ CRITICAL: Install Git LFS First
This repository uses **Git Large File Storage (LFS)** to manage large pre-trained model weights (`.pth` files). If you run `git clone` without having Git LFS installed, you will only download 130-byte text pointer files instead of the actual GB-sized models. This will cause a mysterious `_pickle.UnpicklingError: invalid load key, 'v'` error during inference.

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

## 1. Clone the Repository and Pull Weights
**After** installing Git LFS, clone the repository and pull the actual model weights.

```Bash
git clone [https://github.com/shoda6/DeMemSeg.git](https://github.com/shoda6/DeMemSeg.git)
cd DeMemSeg

# Download the actual heavy weight files
git lfs pull
```
(Verify this worked by checking the file sizes in your model directories; they should be hundreds of MBs, not a few bytes).

## 2. Create a Virtual Environment and Install PyTorch
Create and activate a Python virtual environment (using Conda, venv, or pyenv). Then, install a version of PyTorch that matches your local CUDA installation.

```Bash
# Example using Conda/Mamba
mamba create -n mmdet python=3.10
mamba activate mmdet

# Install PyTorch (Check [https://pytorch.org/](https://pytorch.org/) for the exact command for your CUDA version)
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 3. The Bottleneck: Installing MMCV
MMDetection (v3.3.0) specifically requires MMCV versions >= 2.0.0rc4 and < 2.2.0. Installing the latest MMCV (v2.2.0) will result in an AssertionError.

Recommended Method: Pre-built Binaries via OpenMIM
The easiest way to install MMCV without compilation nightmares is using OpenMIM.

```Bash
pip install -U openmim
mim install "mmcv==2.1.0"
```

### Troubleshooting: Building MMCV from Source
If OpenMIM cannot find a pre-built binary matching your ultra-modern PyTorch/CUDA setup, you must build it from source. You will likely encounter three major traps:
1. CUDA Path Mismatch: The compiler (nvcc) cannot be found.
2. Setuptools Updates: Newer setuptools (v70+) removed pkg_resources, causing ModuleNotFoundError.
3. Build Isolation: pip automatically creates an isolated environment with the newest setuptools, breaking the build again.

### How to safely build from source:

```Bash
# 1. Downgrade setuptools to avoid 'pkg_resources' errors
pip install "setuptools<70" --force-reinstall

# 2. Point to your specific CUDA installation path (Change 12.8 to your version)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Clone MMCV, checkout v2.1.0, and install WITHOUT build isolation
cd ~
git clone [https://github.com/open-mmlab/mmcv.git](https://github.com/open-mmlab/mmcv.git)
cd mmcv
git checkout v2.1.0

# The --no-build-isolation flag is crucial here!
pip install -e . -v --no-build-isolation

# Return to your project directory
cd ~/DeMemSeg
```

## 4. Install MMDetection and Other Dependencies
Once MMCV is successfully installed, install the local mmdetection package. You must use --no-build-isolation here as well to prevent the setuptools trap.

```Bash
# Move into the mmdetection submodule/directory
cd mmdetection

# Install mmdet in editable mode safely
pip install -e . -v --no-build-isolation

# Return to the root directory
cd ..

# Install any remaining requirements
pip install -r requirements.txt
```

## 5. Inference Workaround: PyTorch 2.6+ Security Feature
Starting with PyTorch 2.6, torch.load defaults to weights_only=True for security reasons. Because our models might contain legacy serialization structures, this will trigger a Weights only load failed UnpicklingError.

To bypass this safely (since you downloaded these weights from a trusted source), you must set an environment variable before running your inference scripts.

```Bash
# Disable the forced weights_only check
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

# Now run your script
python script/mmdetection_psm.py
(Alternatively, you can add the following two lines to the very top of your Python scripts, before any other imports):

```Python
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
```
🎉 Setup Complete! You are now ready to run the DeMemSeg pipeline locally.