import redis
import pickle
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = 300  # Cache expiry in seconds (5 minutes)

# Initialize Redis connection
cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

def cache_set(key, value):
    """
    Stores a value in Redis cache.
    
    Args:
        key (str): Cache key.
        value (any): Value to store.
    """
    serialized_value = pickle.dumps(value)
    cache.setex(key, CACHE_TTL, serialized_value)

def cache_get(key):
    """
    Retrieves a value from Redis cache.
    
    Args:
        key (str): Cache key.
    
    Returns:
        Any: The cached value if it exists, otherwise None.
    """
    serialized_value = cache.get(key)
    return pickle.loads(serialized_value) if serialized_value else None

def cache_delete(key):
    """
    Deletes a key from Redis cache.
    
    Args:
        key (str): Cache key.
    """
    cache.delete(key)
