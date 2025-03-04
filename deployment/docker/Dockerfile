# 🚀 JC1 Inference API - Dockerfile
# Using NVIDIA PyTorch base image with CUDA for GPU support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies (DeepSpeed, vLLM, Triton)
RUN pip3 install deepspeed vllm[triton] fastapi uvicorn pydantic

# Copy application files
COPY . .

# Expose API port
EXPOSE 8000

# Start API server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
