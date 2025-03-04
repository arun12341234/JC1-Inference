import time
from app.models.loader import load_model
from app.models.inference import run_inference

def test_model_latency():
    """Measure model inference time (latency)."""
    model = load_model()
    input_text = "Explain Quantum Entanglement in simple terms."

    start_time = time.time()
    output = run_inference(model, input_text)
    end_time = time.time()

    latency = end_time - start_time
    print(f"Model Latency: {latency:.3f}s")

    assert latency < 1.5  # Ensure inference is under 1.5 seconds

def test_model_throughput():
    """Check how many requests the model can process in parallel."""
    model = load_model()
    inputs = ["Tell me a joke", "Translate 'Hello' to French", "Explain AI"]
    
    start_time = time.time()
    results = [run_inference(model, text) for text in inputs]
    end_time = time.time()

    throughput = len(inputs) / (end_time - start_time)
    print(f"Model Throughput: {throughput:.2f} requests/sec")

    assert throughput > 5  # Ensure throughput is at least 5 requests/sec
