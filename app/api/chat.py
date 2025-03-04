from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.loader import load_model
from app.models.tokenizer import tokenize_input
from app.utils.logger import log_request
from app.utils.memory import retrieve_context, store_context

# Initialize router
router = APIRouter()

# Load model once and reuse it
model, tokenizer = load_model()

class ChatRequest(BaseModel):
    user_id: str  # Unique ID to maintain session memory
    message: str  # User input message
    history: list = []  # Previous conversation context
    max_tokens: int = 200  # Token limit for response
    temperature: float = 0.7  # Sampling temperature

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Handles chat requests and generates model responses."""
    try:
        # Log request
        log_request(user_id=request.user_id, message=request.message)
        
        # Retrieve memory context (previous messages)
        history = retrieve_context(request.user_id)
        history.append({"role": "user", "content": request.message})
        
        # Tokenize input
        tokenized_input = tokenize_input(history, tokenizer)
        
        # Generate response from model
        output = model.generate(
            inputs=tokenized_input,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        # Decode response
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Store updated history
        history.append({"role": "assistant", "content": response_text})
        store_context(request.user_id, history)
        
        return {"response": response_text, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
