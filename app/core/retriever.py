"""
retriever.py - Implements Retrieval-Augmented Generation (RAG)
---------------------------------------------------------------
🔹 Features:
- Retrieves relevant context for better LLM responses
- Supports keyword + semantic search (Hybrid RAG)
- Uses FAISS for fast vector retrieval
- Integrates with external knowledge sources

📌 Dependencies:
- FAISS (Vector DB)
- SentenceTransformers (Embeddings)
- SQLite (Metadata storage)
"""

import os
import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

### 🔧 CONFIGURATION ###
DB_PATH = "data/embeddings/document_index"
METADATA_DB = "data/embeddings/document_metadata.db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)


### 📂 DOCUMENT RETRIEVER CLASS ###
class DocumentRetriever:
    """
    Implements Hybrid RAG: FAISS-based vector retrieval + keyword search.
    """

    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)  # L2 Distance Index
        self.metadata_conn = sqlite3.connect(METADATA_DB)
        self._setup_db()

    def _setup_db(self):
        """Initialize metadata database for document storage."""
        cursor = self.metadata_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                vector BLOB,
                source TEXT
            )
            """
        )
        self.metadata_conn.commit()

    def add_document(self, text: str, source: str):
        """
        Stores document & vector embedding in FAISS.
        """
        vector = embedder.encode([text])[0].astype(np.float32)
        self.index.add(np.array([vector]))  # Add vector to FAISS
        cursor = self.metadata_conn.cursor()
        cursor.execute(
            "INSERT INTO documents (text, vector, source) VALUES (?, ?, ?)",
            (text, vector.tobytes(), source),
        )
        self.metadata_conn.commit()

    def retrieve_documents(self, query: str, top_k=5) -> List[str]:
        """
        Retrieves top-k relevant documents for a given query.
        """
        query_vector = embedder.encode([query])[0].astype(np.float32)
        D, I = self.index.search(np.array([query_vector]), top_k)  # FAISS search
        results = []
        cursor = self.metadata_conn.cursor()
        for idx in I[0]:
            if idx == -1:
                continue  # No match found
            cursor.execute("SELECT text FROM documents WHERE id=?", (idx + 1,))
            result = cursor.fetchone()
            if result:
                results.append(result[0])
        return results

    def hybrid_search(self, query: str, top_k=5) -> List[str]:
        """
        Hybrid search combining FAISS vector search with keyword-based filtering.
        """
        query_vector = embedder.encode([query])[0].astype(np.float32)
        D, I = self.index.search(np.array([query_vector]), top_k)  # FAISS search

        # Retrieve vector-based results
        vector_results = []
        cursor = self.metadata_conn.cursor()
        for idx in I[0]:
            if idx == -1:
                continue
            cursor.execute("SELECT text FROM documents WHERE id=?", (idx + 1,))
            result = cursor.fetchone()
            if result:
                vector_results.append(result[0])

        # Retrieve keyword-based results
        keyword_results = []
        cursor.execute(
            "SELECT text FROM documents WHERE text LIKE ? LIMIT ?",
            (f"%{query}%", top_k),
        )
        keyword_results = [row[0] for row in cursor.fetchall()]

        # Combine results
        combined_results = list(set(vector_results + keyword_results))
        return combined_results[:top_k]

    def clear_documents(self):
        """Clears stored documents."""
        self.index.reset()
        cursor = self.metadata_conn.cursor()
        cursor.execute("DELETE FROM documents")
        self.metadata_conn.commit()


### 🛠️ EXAMPLE USAGE ###
if __name__ == "__main__":
    retriever = DocumentRetriever()

    # Store documents
    retriever.add_document("Python is a versatile programming language.", "Wikipedia")
    retriever.add_document("AI models are transforming industries.", "Research Paper")

    # Retrieve relevant documents
    results = retriever.retrieve_documents("Tell me about Python")
    print(f"RAG Retrieval Results: {results}")
