#!/bin/bash
# 🚀 JC1 Setup Script - Automates Environment Setup

echo "🔹 Initializing JC1 Environment Setup..."

# Step 1: Update & Install Required Packages
echo "🔹 Updating system and installing dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential software-properties-common curl wget git

# Step 2: Install Python and Dependencies
echo "🔹 Installing Python 3.10 & pip..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Step 3: Create Virtual Environment
echo "🔹 Creating Python virtual environment..."
python3.10 -m venv jc1_env
source jc1_env/bin/activate

# Step 4: Upgrade pip and Install Required Python Libraries
echo "🔹 Installing required Python libraries..."
pip install --upgrade pip
pip install fastapi uvicorn numpy torch torchvision torchaudio transformers deepspeed vllm tritonclient[all] pydantic

# Step 5: Install NVIDIA Drivers & CUDA (Ensure compatibility with H100 GPUs)
echo "🔹 Installing NVIDIA Drivers, CUDA & cuDNN..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2 libcudnn8

# Step 6: Install NVIDIA Triton Inference Server
echo "🔹 Installing NVIDIA Triton Inference Server..."
docker pull nvcr.io/nvidia/tritonserver:23.12-py3
mkdir -p /models/triton-repository

# Step 7: Clone JC1 Repository & Setup Configuration
echo "🔹 Cloning JC1 Repository..."
git clone https://github.com/aryantechworld/JC1-Inference.git
cd JC1-Inference

# Step 8: Set Up Environment Variables
echo "🔹 Configuring environment variables..."
export JC1_MODEL_PATH="/models/triton-repository"
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 for inference
export PYTHONPATH=$(pwd):$PYTHONPATH

# Step 9: Start Triton Inference Server
echo "🔹 Starting Triton Inference Server..."
docker run --rm --gpus all -d \
    --name jc1_triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /models/triton-repository:/models \
    nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models

# Step 10: Final Check & Start FastAPI Server
echo "🔹 Running final checks..."
python3 -m app.main

echo "✅ JC1 Environment Setup Complete!"
