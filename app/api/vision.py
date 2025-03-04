"""
vision.py - Multimodal Image Processing API for JC1
---------------------------------------------------
🔹 Features:
- Image captioning (describe images using LLM)
- Visual Question Answering (VQA)
- OCR (Text extraction from images)

📌 Dependencies:
- transformers (for vision-language models)
- fastapi (for API handling)
- PIL (image processing)
"""

import io
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor
from app.utils.logger import logger

# Initialize API router for vision processing
router = APIRouter()

# Load pre-trained vision-language models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image Captioning Model
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# OCR Model
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
ocr_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")


def process_image(image: UploadFile):
    """
    Converts uploaded image into PIL format.
    """
    try:
        image_data = Image.open(io.BytesIO(image.file.read()))
        return image_data
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format.")


def generate_caption(image: Image.Image):
    """
    Generates a caption for the given image using BLIP.
    """
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs)
    caption = caption_processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption


def extract_text(image: Image.Image):
    """
    Performs OCR (Optical Character Recognition) on the given image.
    """
    pixel_values = ocr_processor(image, return_tensors="pt").pixel_values.to(device)
    output = ocr_model.generate(pixel_values)
    extracted_text = ocr_processor.batch_decode(output, skip_special_tokens=True)[0]
    return extracted_text


@router.post("/vision/caption/")
async def api_generate_caption(image: UploadFile = File(...)):
    """
    API Endpoint: Generates a descriptive caption for an uploaded image.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        dict: Generated caption.
    """
    image_data = process_image(image)
    caption = generate_caption(image_data)
    return {"status": "success", "caption": caption}


@router.post("/vision/ocr/")
async def api_extract_text(image: UploadFile = File(...)):
    """
    API Endpoint: Extracts text from an uploaded image using OCR.

    Args:
        image (UploadFile): The uploaded image file.

    Returns:
        dict: Extracted text.
    """
    image_data = process_image(image)
    extracted_text = extract_text(image_data)
    return {"status": "success", "extracted_text": extracted_text}
