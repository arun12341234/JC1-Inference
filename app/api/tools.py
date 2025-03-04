"""
tools.py - External Tool Integrations for JC1
---------------------------------------------------
🔹 Features:
- Web Search (Google/Bing API, Local Knowledge Index)
- Retrieval-Augmented Generation (RAG) for document retrieval
- External Plugin Integrations (e.g., Wolfram Alpha, Wikipedia)

📌 Dependencies:
- requests (for API calls)
- transformers (if integrating an LLM-based retriever)
- FAISS (for RAG memory store)
"""

import os
import requests
from fastapi import APIRouter, HTTPException, Query
from app.core.retriever import retrieve_documents
from app.utils.logger import logger

# Initialize API router for tool integrations
router = APIRouter()

# Load environment variables for search API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
BING_API_KEY = os.getenv("BING_API_KEY")


def google_search(query: str, num_results: int = 5):
    """
    Performs a Google Search using Google Custom Search API.

    Args:
        query (str): Search query.
        num_results (int): Number of results to return.

    Returns:
        list: Top search results with titles & URLs.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logger.error("Google API Key or CX missing")
        return []

    search_url = f"https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "num": num_results}
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [{"title": r["title"], "link": r["link"]} for r in results]
    
    except requests.RequestException as e:
        logger.error(f"Google search failed: {e}")
        return []


def bing_search(query: str, num_results: int = 5):
    """
    Performs a Bing Search using Bing Search API.

    Args:
        query (str): Search query.
        num_results (int): Number of results to return.

    Returns:
        list: Top search results with titles & URLs.
    """
    if not BING_API_KEY:
        logger.error("Bing API Key missing")
        return []

    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": num_results}
    
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("webPages", {}).get("value", [])
        return [{"title": r["name"], "link": r["url"]} for r in results]
    
    except requests.RequestException as e:
        logger.error(f"Bing search failed: {e}")
        return []


@router.get("/search/")
async def search_tool(
    query: str = Query(..., description="Search query"),
    engine: str = Query("google", description="Search engine (google/bing)"),
    num_results: int = Query(5, description="Number of search results")
):
    """
    API Endpoint: Perform a web search using Google or Bing.

    Args:
        query (str): The search query.
        engine (str): Search engine (google or bing).
        num_results (int): Number of results to return.

    Returns:
        dict: Search results.
    """
    try:
        if engine.lower() == "google":
            results = google_search(query, num_results)
        elif engine.lower() == "bing":
            results = bing_search(query, num_results)
        else:
            raise HTTPException(status_code=400, detail="Invalid search engine")

        return {"query": query, "results": results}
    
    except Exception as e:
        logger.error(f"Search API failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/retrieve/")
async def retrieve_knowledge(query: str = Query(..., description="Query for document retrieval")):
    """
    API Endpoint: Retrieve relevant documents using RAG.

    Args:
        query (str): The search query.

    Returns:
        dict: Retrieved document content.
    """
    try:
        retrieved_docs = retrieve_documents(query)
        return {"query": query, "retrieved_documents": retrieved_docs}
    
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Knowledge retrieval failed")
