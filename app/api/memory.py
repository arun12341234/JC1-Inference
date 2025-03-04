"""
memory.py - Context Memory Retrieval for JC1
---------------------------------------------------
🔹 Features:
- Stores chat history in a vector database
- Retrieves relevant past interactions
- Supports multi-turn conversations for better context memory

📌 Dependencies:
- chromadb (for vector storage)
- transformers (tokenization utilities)
"""

import os
import chromadb
from fastapi import APIRouter, HTTPException, Query
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger

# Initialize API router for memory management
router = APIRouter()

# Load environment variable for memory storage path
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory_db")

# Initialize ChromaDB client (Vector Database for memory storage)
chroma_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
memory_collection = chroma_client.get_or_create_collection("jc1_memory")

# Load embedding model (used for vectorizing text)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def store_memory(user_id: str, conversation: str):
    """
    Stores a conversation in the vector database.

    Args:
        user_id (str): Unique identifier for the user.
        conversation (str): Chat message to store.

    Returns:
        str: Confirmation message.
    """
    embedding = embedding_model.encode(conversation).tolist()

    try:
        memory_collection.add(
            ids=[user_id + "-" + str(len(memory_collection.get()["ids"]))],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id, "text": conversation}],
        )
        return "Conversation stored successfully."
    
    except Exception as e:
        logger.error(f"Memory storage failed: {e}")
        return "Error storing conversation."


def retrieve_memory(user_id: str, query: str, top_k: int = 3):
    """
    Retrieves relevant past conversations using vector similarity search.

    Args:
        user_id (str): Unique identifier for the user.
        query (str): Query to match against past conversations.
        top_k (int): Number of most relevant results to return.

    Returns:
        list: Retrieved memory fragments.
    """
    embedding = embedding_model.encode(query).tolist()

    try:
        results = memory_collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        return [res["text"] for res in results["metadatas"][0]]
    
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return []


@router.post("/memory/store/")
async def api_store_memory(user_id: str, conversation: str):
    """
    API Endpoint: Stores user conversation memory.

    Args:
        user_id (str): The unique identifier for the user.
        conversation (str): The message to store.

    Returns:
        dict: Success message.
    """
    message = store_memory(user_id, conversation)
    return {"status": "success", "message": message}


@router.get("/memory/retrieve/")
async def api_retrieve_memory(user_id: str, query: str, top_k: int = 3):
    """
    API Endpoint: Retrieves past conversation memory based on similarity search.

    Args:
        user_id (str): The unique identifier for the user.
        query (str): The query to search in memory.
        top_k (int): Number of past interactions to retrieve.

    Returns:
        dict: Retrieved conversation fragments.
    """
    retrieved_memories = retrieve_memory(user_id, query, top_k)
    if not retrieved_memories:
        raise HTTPException(status_code=404, detail="No relevant memory found.")
    
    return {"status": "success", "retrieved_memories": retrieved_memories}
