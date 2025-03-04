import pytest
from app.models.loader import load_model
from app.models.inference import run_inference

@pytest.fixture(scope="module")
def model():
    """Load the AI model before tests."""
    return load_model()

def test_model_inference(model):
    """Check if the model correctly generates text output."""
    input_text = "What is the capital of France?"
    output = run_inference(model, input_text)
    
    assert isinstance(output, str)
    assert len(output) > 0
    assert "Paris" in output
