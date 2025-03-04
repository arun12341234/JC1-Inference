"""
logger.py - Logging Utility
----------------------------
🔹 Features:
- Configures logging for console and file output
- Uses rotating file handler for log size management
- Supports structured JSON logging for better analysis

📌 Dependencies:
- logging (built-in Python module)
- os (for file path handling)
"""

import logging
import os
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# Load log path from config or set default
LOG_DIR = os.getenv("LOG_PATH", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """
    Custom JSON log formatter.
    Converts log messages into structured JSON format.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        return json.dumps(log_record, ensure_ascii=False)

# Configure Logger
def get_logger(name="JC1-Logger"):
    """
    Returns a configured logger instance.

    🔹 Features:
    - Logs to both **console** and **file**
    - Uses a **rotating file handler** (max 5MB per file, keeps 3 backups)
    - Supports **JSON-formatted logs** for structured analysis
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set global log level

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # File Handler with Rotation
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    # Attach Handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Initialize Logger
logger = get_logger()
logger.info("Logger initialized successfully.")
