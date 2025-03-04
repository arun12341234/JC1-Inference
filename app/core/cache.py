"""
cache.py - Implements Key-Value Caching & Session Handling
------------------------------------------------------------
🔹 Features:
- Stores temporary inference responses for faster LLM interactions
- Uses Redis (optional) for persistent caching across sessions
- Handles automatic expiration of cache entries
- Supports multi-turn conversation memory

📌 Dependencies:
- Redis (for distributed caching)
- Pickle (for serialization)
"""

import os
import time
import pickle
import redis
from typing import Optional

### 🔧 CONFIGURATION ###
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = 300  # Cache expiry in seconds (5 minutes)

# Try connecting to Redis if available
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
    redis_available = redis_client.ping()
except redis.ConnectionError:
    redis_available = False


### 📂 CACHE MANAGER CLASS ###
class CacheManager:
    """
    Handles key-value caching for inference responses and session data.
    """

    def __init__(self):
        self.local_cache = {}  # Fallback dictionary if Redis is unavailable
        if redis_available:
            print(f"✅ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        else:
            print("⚠️ Redis not available, falling back to in-memory caching.")

    def set(self, key: str, value: dict, ttl: Optional[int] = CACHE_TTL):
        """
        Stores a value in cache (either Redis or local).
        """
        serialized_value = pickle.dumps(value)
        if redis_available:
            redis_client.setex(key, ttl, serialized_value)
        else:
            self.local_cache[key] = (serialized_value, time.time() + ttl)

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieves a value from cache.
        """
        if redis_available:
            cached_value = redis_client.get(key)
            return pickle.loads(cached_value) if cached_value else None
        else:
            if key in self.local_cache:
                value, expiry = self.local_cache[key]
                if time.time() < expiry:
                    return pickle.loads(value)
                else:
                    del self.local_cache[key]  # Remove expired entry
        return None

    def delete(self, key: str):
        """
        Deletes a key from the cache.
        """
        if redis_available:
            redis_client.delete(key)
        elif key in self.local_cache:
            del self.local_cache[key]

    def clear_cache(self):
        """
        Clears all cache entries.
        """
        if redis_available:
            redis_client.flushdb()
        else:
            self.local_cache.clear()


### 🛠️ EXAMPLE USAGE ###
if __name__ == "__main__":
    cache = CacheManager()

    # Store session data
    session_data = {"conversation": ["Hello, how can I help you?", "What is AI?"]}
    cache.set("session_123", session_data)

    # Retrieve session data
    retrieved_data = cache.get("session_123")
    print(f"Retrieved Session Data: {retrieved_data}")

    # Clear cache
    cache.clear_cache()
