import faiss
import numpy as np
import os
import pickle

EMBEDDINGS_DIR = "data/embeddings"
INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")

def save_embeddings(vectors, metadata):
    """
    Saves embeddings to FAISS index and metadata to a pickle file.
    
    Args:
        vectors (np.array): NumPy array of vector embeddings.
        metadata (list): List of associated metadata (e.g., document text).
    """
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    d = vectors.shape[1]  # Dimensionality of vectors
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def load_embeddings():
    """
    Loads FAISS index and metadata.
    
    Returns:
        index (faiss.IndexFlatL2): FAISS index object.
        metadata (list): List of metadata associated with embeddings.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def search(query_vector, k=5):
    """
    Searches the FAISS index for nearest neighbors.
    
    Args:
        query_vector (np.array): Query embedding.
        k (int): Number of nearest neighbors to return.
    
    Returns:
        List of closest matching metadata entries.
    """
    index, metadata = load_embeddings()
    if index is None:
        return []

    distances, indices = index.search(np.array([query_vector]), k)
    return [metadata[i] for i in indices[0] if i >= 0]
