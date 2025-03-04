"""
inference.py - Optimized Model Inference Execution
---------------------------------------------------
🔹 Features:
- Batched inference execution
- Uses DeepSpeed & vLLM for optimized GPU inference
- Handles tokenization, decoding, and post-processing
- Async execution for handling multiple requests

📌 Dependencies:
- transformers (Hugging Face model loader)
- torch (PyTorch backend)
- deepspeed (GPU acceleration)
- vllm (Faster LLM inference)
"""

import os
import torch
import deepspeed
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.utils.logger import logger
from app.models.tokenizer import tokenizer  # Import our tokenizer utility

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B")
USE_DEEPSPEED = os.getenv("USE_DEEPSPEED", "true").lower() == "true"
USE_VLLM = os.getenv("USE_VLLM", "true").lower() == "true"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))  # Limit token generation

class ModelInference:
    """
    Handles optimized model inference using DeepSpeed, vLLM, or native PyTorch.
    """

    def __init__(self):
        """
        Initializes model based on selected backend.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if USE_VLLM:
            self.init_vllm()
        elif USE_DEEPSPEED:
            self.init_deepspeed()
        else:
            self.init_torch()

    def init_vllm(self):
        """
        Initializes vLLM for faster inference.
        """
        try:
            self.model = vllm.LLM(MODEL_NAME, tensor_parallel_size=torch.cuda.device_count())
            logger.info("✅ Loaded model with vLLM for optimized inference.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM: {e}")
            raise RuntimeError("vLLM loading failed.")

    def init_deepspeed(self):
        """
        Initializes DeepSpeed for efficient model execution.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
            )
            self.model = deepspeed.init_inference(self.model, dtype=torch.float16)
            logger.info("✅ Loaded model with DeepSpeed for efficient execution.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DeepSpeed: {e}")
            raise RuntimeError("DeepSpeed initialization failed.")

    def init_torch(self):
        """
        Initializes model using basic PyTorch inference.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)
            logger.info("✅ Loaded model with PyTorch (basic execution).")
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
            raise RuntimeError("PyTorch model loading failed.")

    def generate_response(self, input_text):
        """
        Generates model output for a given input text.
        """
        tokens = tokenizer.encode(input_text)
        input_tensor = torch.tensor([tokens]).to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(input_tensor, max_new_tokens=MAX_NEW_TOKENS)
        
        return tokenizer.decode(output_tokens[0].tolist())

    async def async_generate_response(self, input_text):
        """
        Async version of generate_response for parallel request handling.
        """
        return self.generate_response(input_text)


# Instantiate global inference handler
model_inference = ModelInference()
