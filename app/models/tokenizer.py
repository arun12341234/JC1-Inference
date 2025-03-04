"""
tokenizer.py - Tokenization Utilities for LLM
-----------------------------------------------
🔹 Features:
- Supports different tokenization methods (BPE, WordPiece, SentencePiece)
- Loads tokenizers dynamically based on model configuration
- Handles special token processing (padding, truncation)
- Optimized for batch inference

📌 Dependencies:
- transformers (Hugging Face tokenizer)
- sentencepiece (LLM-compatible tokenization)
"""

import os
from transformers import AutoTokenizer
import sentencepiece as spm
from app.utils.logger import logger

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8B")
TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "auto")  # Options: auto, bpe, wordpiece, sentencepiece
MAX_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH", 2048))  # Adjust for longer contexts

class Tokenizer:
    """
    Tokenization wrapper for different models.
    """

    def __init__(self):
        self.tokenizer = None
        if TOKENIZER_TYPE == "bpe":
            self.load_bpe_tokenizer()
        elif TOKENIZER_TYPE == "wordpiece":
            self.load_wordpiece_tokenizer()
        elif TOKENIZER_TYPE == "sentencepiece":
            self.load_sentencepiece_tokenizer()
        else:
            self.load_auto_tokenizer()

    def load_auto_tokenizer(self):
        """
        Loads default Hugging Face tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logger.info(f"✅ Loaded tokenizer for {MODEL_NAME} (AutoTokenizer).")
        except Exception as e:
            logger.error(f"❌ Failed to load AutoTokenizer: {e}")
            raise RuntimeError("Tokenizer loading failed.")

    def load_bpe_tokenizer(self):
        """
        Loads a Byte Pair Encoding (BPE) tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            logger.info(f"✅ Loaded BPE tokenizer for {MODEL_NAME}.")
        except Exception as e:
            logger.error(f"❌ Failed to load BPE tokenizer: {e}")
            raise RuntimeError("BPE Tokenizer loading failed.")

    def load_wordpiece_tokenizer(self):
        """
        Loads a WordPiece tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            logger.info(f"✅ Loaded WordPiece tokenizer for {MODEL_NAME}.")
        except Exception as e:
            logger.error(f"❌ Failed to load WordPiece tokenizer: {e}")
            raise RuntimeError("WordPiece Tokenizer loading failed.")

    def load_sentencepiece_tokenizer(self):
        """
        Loads a SentencePiece tokenizer for LLMs.
        """
        try:
            self.tokenizer = spm.SentencePieceProcessor(model_file=f"{MODEL_NAME}/sentencepiece.model")
            logger.info(f"✅ Loaded SentencePiece tokenizer for {MODEL_NAME}.")
        except Exception as e:
            logger.error(f"❌ Failed to load SentencePiece tokenizer: {e}")
            raise RuntimeError("SentencePiece Tokenizer loading failed.")

    def encode(self, text):
        """
        Encodes input text into token IDs.
        """
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return self.tokenizer.encode(text, out_type=int)
        return self.tokenizer.encode(text, truncation=True, padding=True, max_length=MAX_LENGTH)

    def decode(self, tokens):
        """
        Decodes token IDs back into text.
        """
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return self.tokenizer.decode(tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# Instantiate global tokenizer
tokenizer = Tokenizer()
