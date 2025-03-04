"""
config.py - Application Configuration
--------------------------------------
🔹 Features:
- Centralizes API keys, model paths, and optimization settings
- Uses dotenv for environment-based configuration
- Provides structured validation via Pydantic

📌 Dependencies:
- python-dotenv (loads .env variables)
- pydantic (validates config schema)
"""

import os
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class AppConfig(BaseSettings):
    """
    Application Configuration Model using Pydantic.
    """
    # Model Configurations
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B")
    USE_VLLM: bool = os.getenv("USE_VLLM", "true").lower() == "true"
    USE_DEEPSPEED: bool = os.getenv("USE_DEEPSPEED", "true").lower() == "true"
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 4096))

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-default-key")
    VECTOR_DB_API_KEY: str = os.getenv("VECTOR_DB_API_KEY", "your-vector-db-key")

    # Security Configurations
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    AUTH_SECRET: str = os.getenv("AUTH_SECRET", "supersecretkey")

    # Paths
    DATA_PATH: str = os.getenv("DATA_PATH", "./data")
    LOG_PATH: str = os.getenv("LOG_PATH", "./logs")
    CACHE_PATH: str = os.getenv("CACHE_PATH", "./data/cache")

    class Config:
        """
        Pydantic settings configuration.
        """
        env_file = ".env"
        case_sensitive = True


# Initialize Configuration
config = AppConfig()
