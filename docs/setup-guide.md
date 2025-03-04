# 🚀 JC1-Inference Setup Guide

## 📌 1️⃣ Prerequisites
- **Python 3.10+**
- **CUDA 11.8+ (for GPU acceleration)**
- **Docker & Docker Compose**
- **Kubernetes (optional for large-scale deployment)**
- **NVIDIA Triton Inference Server** (for high-speed inference)

## 📌 2️⃣ Installation

### 🔹 Option 1: Local Installation
```bash
git clone https://github.com/your-repo/jc1-inference.git
cd jc1-inference
pip install -r requirements.txt
```
🔹 Option 2: Run with Docker
```bash
docker-compose up --build
```
🔹 Option 3: Run on Kubernetes
```bash
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
```
📌 3️⃣ Running the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
The API will now be available at: http://localhost:8000/api/v1/