#!/bin/bash
# ==========================================
# Setup Script for FastVLM
#
# Use runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 as base image
# ==========================================

echo "🔧 Creating workspace..."

cd ..
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
# source ~/.bashrc
source $HOME/miniconda/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd fastvlm-adaptation/

conda create -n fastvlm python=3.10
conda activate fastvlm
pip install --upgrade pip
pip install -e .
pip install decord deepspeed matplotlib wandb evaluate bert-score nltk

apt install unzip

bash get_models.sh

export PYTHONPATH="./:$PYTHONPATH"

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

which nvcc

python -c "import torch; print(f'torch version: {torch.__version__}')"

echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "❌ nvcc not found"
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ PyTorch cannot see CUDA')
"

echo ""

apt-get update
pip install openpyxl scikit-learn sentence-transformers
apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super


echo "✅ Setup complete!"
echo "🚀 FastVLM environment is ready."

# Initialize WandB
echo "🔑 Logging into WandB..."
wandb login

# Initialize HuggingFace Hub
echo "🤗 Logging into HuggingFace Hub..."
hf auth login

source ~/.bashrc
