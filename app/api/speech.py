"""
speech.py - Speech-to-Text (ASR) API for JC1-Inference
--------------------------------------------------------
This module provides an API for converting speech/audio inputs into text
using an Automatic Speech Recognition (ASR) model.

🔹 Features:
- Accepts audio files (MP3, WAV, FLAC, OGG).
- Uses a pre-trained ASR model (like Whisper, Wav2Vec2, or DeepSpeech).
- Returns transcribed text.

📌 Dependencies:
- FastAPI: API framework
- Pydantic: Request validation
- Python-Torchaudio or Whisper for ASR model integration
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torchaudio
from app.models.loader import load_asr_model
from app.utils.logger import logger

# Initialize FastAPI router for Speech-to-Text API
router = APIRouter()

# Load ASR model (Whisper, Wav2Vec2, DeepSpeech, etc.)
asr_model = load_asr_model()

# Supported audio formats
ALLOWED_AUDIO_FORMATS = {"audio/wav", "audio/mp3", "audio/flac", "audio/ogg"}

def preprocess_audio(audio_path: str):
    """
    Preprocesses an audio file for the ASR model.
    
    - Loads the audio file using torchaudio.
    - Converts to the required sample rate (16kHz default).
    - Normalizes the waveform.
    
    Args:
        audio_path (str): Path to the uploaded audio file.
    
    Returns:
        torch.Tensor: Processed waveform
        int: Sample rate
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        return waveform, sample_rate

    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise HTTPException(status_code=500, detail="Audio preprocessing failed")


@router.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    """
    API Endpoint: Converts speech/audio input to text.
    
    - Accepts an audio file (MP3, WAV, FLAC, OGG).
    - Processes the audio into a waveform.
    - Runs inference on the ASR model.
    - Returns transcribed text.
    
    Args:
        file (UploadFile): The uploaded audio file.

    Returns:
        dict: Transcription result
    """
    try:
        # Validate file format
        if file.content_type not in ALLOWED_AUDIO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.content_type}")

        # Save uploaded file temporarily
        temp_audio_path = f"temp_audio/{file.filename}"
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(file.file.read())

        # Preprocess audio file
        waveform, sample_rate = preprocess_audio(temp_audio_path)

        # Run ASR inference
        transcription = asr_model.transcribe(waveform, sample_rate)

        return {"filename": file.filename, "transcription": transcription}

    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process speech")

