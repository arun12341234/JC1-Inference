"""
main.py - JC1 Inference API (FastAPI Entry Point)
-------------------------------------------------
🔹 Features:
- Serves LLM inference endpoints (Chat, Vision, Speech)
- Uses FastAPI for high-performance async API handling
- Implements Cross-Origin Resource Sharing (CORS)
- Includes logging and authentication middleware

📌 Dependencies:
- `fastapi` → For API handling
- `uvicorn` → For ASGI server
- `pydantic` → For request validation
- `app.api.chat`, `app.api.vision`, `app.api.speech` → API modules
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import API endpoints
from app.api.chat import router as chat_router
from app.api.vision import router as vision_router
from app.api.speech import router as speech_router

# Initialize FastAPI App
app = FastAPI(
    title="JC1 Inference API",
    description="🚀 High-performance API for LLM, Vision, and Speech inference",
    version="1.0.0",
)

### 🌍 CORS Middleware ###
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### 🔗 Include API Routes ###
app.include_router(chat_router, prefix="/api/chat", tags=["Chat LLM"])
app.include_router(vision_router, prefix="/api/vision", tags=["Vision Processing"])
app.include_router(speech_router, prefix="/api/speech", tags=["Speech-to-Text"])

### 📍 Root Endpoint ###
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "🚀 JC1 Inference API is running!"}

### 🚀 Run API ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
