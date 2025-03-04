"""
memory.py - Vector Database for Context Memory (RAG)
------------------------------------------------------
🔹 Features:
- Stores & retrieves conversation context using FAISS
- Supports long-term memory for chat-based AI assistants
- Uses vector embeddings for fast semantic search
- Integrates with LLM to maintain conversation history

📌 Dependencies:
- FAISS (for vector search)
- SentenceTransformers (for embedding generation)
- SQLite (for metadata storage)
"""

import os
import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

### 🔧 CONFIGURATION ###
DB_PATH = "data/embeddings/memory_index"
METADATA_DB = "data/embeddings/memory_metadata.db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)


### 🧠 MEMORY INDEX SETUP ###
class MemoryDB:
    """
    Manages conversation memory using FAISS vector storage.
    """

    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)  # L2 Distance-based FAISS Index
        self.metadata_conn = sqlite3.connect(METADATA_DB)
        self._setup_db()

    def _setup_db(self):
        """Initialize metadata database for mapping conversations."""
        cursor = self.metadata_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                vector BLOB,
                conversation_id TEXT
            )
            """
        )
        self.metadata_conn.commit()

    def add_memory(self, conversation_id: str, text: str):
        """
        Stores text & its vector embedding in memory.
        """
        vector = embedder.encode([text])[0].astype(np.float32)
        self.index.add(np.array([vector]))  # Add vector to FAISS
        cursor = self.metadata_conn.cursor()
        cursor.execute(
            "INSERT INTO memory (text, vector, conversation_id) VALUES (?, ?, ?)",
            (text, vector.tobytes(), conversation_id),
        )
        self.metadata_conn.commit()

    def retrieve_memory(self, query: str, top_k=5):
        """
        Retrieves top-k most relevant memory entries for a given query.
        """
        query_vector = embedder.encode([query])[0].astype(np.float32)
        D, I = self.index.search(np.array([query_vector]), top_k)  # FAISS search
        results = []
        cursor = self.metadata_conn.cursor()
        for idx in I[0]:
            if idx == -1:
                continue  # No match found
            cursor.execute("SELECT text FROM memory WHERE id=?", (idx + 1,))
            result = cursor.fetchone()
            if result:
                results.append(result[0])
        return results

    def clear_memory(self):
        """Clears all stored conversation memory."""
        self.index.reset()
        cursor = self.metadata_conn.cursor()
        cursor.execute("DELETE FROM memory")
        self.metadata_conn.commit()


### 🛠️ EXAMPLE USAGE ###
if __name__ == "__main__":
    memory = MemoryDB()

    # Store conversation
    memory.add_memory("conv_123", "The capital of France is Paris.")
    memory.add_memory("conv_123", "AI models like GPT-4 are trained on vast datasets.")

    # Retrieve relevant memory
    result = memory.retrieve_memory("What is the capital of France?")
    print(f"Memory Retrieval Result: {result}")
