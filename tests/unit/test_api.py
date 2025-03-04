import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)

def test_chat_endpoint():
    """Test the chat API with a sample input."""
    response = client.post("/api/chat", json={"message": "Hello JC1!"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_vision_endpoint():
    """Test vision API with an image upload."""
    files = {"file": ("test.jpg", b"binary_image_data", "image/jpeg")}
    response = client.post("/api/vision", files=files)
    assert response.status_code == 200
    assert "description" in response.json()

def test_speech_endpoint():
    """Test speech-to-text API."""
    files = {"file": ("test.wav", b"audio_data", "audio/wav")}
    response = client.post("/api/speech", files=files)
    assert response.status_code == 200
    assert "transcript" in response.json()
