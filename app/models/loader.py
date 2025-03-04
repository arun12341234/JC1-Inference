"""
loader.py - Model Loader with DeepSpeed & vLLM (GPU Acceleration)
-----------------------------------------------------------------
🔹 Features:
- Loads model efficiently with vLLM or DeepSpeed
- Supports multi-GPU execution
- Fallbacks to PyTorch if DeepSpeed/vLLM are unavailable
- Handles memory optimization for large models

📌 Dependencies:
- torch (PyTorch backend)
- deepspeed (DeepSpeed for model parallelism)
- vllm (Optimized inference)
- transformers (Hugging Face model loading)
"""

import os
import torch
import deepspeed
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.utils.logger import logger

# Load configurations from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B")
USE_VLLM = os.getenv("USE_VLLM", "true").lower() == "true"
USE_DEEPSPEED = os.getenv("USE_DEEPSPEED", "true").lower() == "true"

class ModelLoader:
    """
    Loads the language model using vLLM, DeepSpeed, or PyTorch fallback.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        if USE_VLLM:
            self.load_vllm()
        elif USE_DEEPSPEED:
            self.load_deepspeed()
        else:
            self.load_torch()

    def load_vllm(self):
        """
        Loads the model using vLLM for ultra-fast inference.
        """
        try:
            self.model = vllm.LLM(MODEL_NAME, tensor_parallel_size=torch.cuda.device_count())
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logger.info("✅ Model loaded with vLLM for high-speed inference.")
        except Exception as e:
            logger.error(f"❌ vLLM loading failed: {e}. Switching to DeepSpeed.")
            self.load_deepspeed()

    def load_deepspeed(self):
        """
        Loads the model using DeepSpeed for optimized execution.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = deepspeed.init_inference(self.model, dtype=torch.float16)
            logger.info("✅ Model loaded with DeepSpeed.")
        except Exception as e:
            logger.error(f"❌ DeepSpeed loading failed: {e}. Switching to PyTorch fallback.")
            self.load_torch()

    def load_torch(self):
        """
        Loads the model using PyTorch as a last-resort fallback.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)
            logger.info("✅ Model loaded using PyTorch fallback.")
        except Exception as e:
            logger.critical(f"❌ PyTorch model loading failed: {e}. Cannot proceed.")

    def get_model(self):
        """
        Returns the loaded model and tokenizer.
        """
        return self.model, self.tokenizer


# Instantiate model loader globally
model_loader = ModelLoader()
