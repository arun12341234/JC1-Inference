import time
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_PATH = "checkpoints/jc1-rlhf"
NUM_SAMPLES = 10  # Number of test queries
BATCH_SIZE = 4  # Batch processing size

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")

# Sample prompts
test_prompts = [
    "What is the capital of France?",
    "Explain quantum entanglement in simple terms.",
    "Translate 'Hello, how are you?' to French.",
    "Write a short poem about space exploration."
]

# Extend prompts to match batch size
test_prompts = test_prompts * (NUM_SAMPLES // len(test_prompts))

# 🚀 Benchmark Function
def benchmark_inference():
    print("🚀 Running inference benchmark...")

    # Tokenize input
    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Warm-up run
    print("🔥 Warming up model...")
    _ = model.generate(**inputs)

    # Measure inference time
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=100)
    end_time = time.time()

    # Decode outputs
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Compute latency
    latency = (end_time - start_time) / len(test_prompts)
    print(f"⚡ Average Latency per Sample: {latency:.4f} seconds")

    return responses

# Run benchmark
if __name__ == "__main__":
    results = benchmark_inference()
    for idx, res in enumerate(results):
        print(f"\nSample {idx+1}: {res}")
