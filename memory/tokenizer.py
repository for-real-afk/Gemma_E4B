import logging

logger = logging.getLogger(__name__)

class TokenCalculator:
    def __init__(self):
        self.gemma_tokenizer = None
        self.tiktoken_encoder = None
        self.initialized = False

    def initialize(self) -> None:
        """Load and cache tokenizers in memory."""
        # 1. Load tiktoken
        try:
            import tiktoken
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("tiktoken encoder 'cl100k_base' loaded successfully.")
        except Exception as e:
            logger.error("Failed to load tiktoken encoder: %s", e)

        # 2. Load AutoTokenizer (transformers)
        try:
            from transformers import AutoTokenizer
            
            # Try loading alpindale/gemma-tokenizer (public/ungated mirror of gemma)
            try:
                self.gemma_tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-tokenizer")
                logger.info("Gemma AutoTokenizer loaded successfully from 'alpindale/gemma-tokenizer'.")
            except Exception as e:
                logger.warning("Failed to load alpindale/gemma-tokenizer: %s. Trying google/gemma-2b...", e)
                try:
                    self.gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
                    logger.info("Gemma AutoTokenizer loaded successfully from 'google/gemma-2b'.")
                except Exception as e2:
                    logger.warning("Failed to load google/gemma-2b: %s. Trying gpt2 as fallback...", e2)
                    try:
                        self.gemma_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        logger.info("AutoTokenizer loaded successfully from 'gpt2' fallback.")
                    except Exception as e3:
                        logger.error("Failed to load all tokenizer options: %s", e3)
        except Exception as e:
            logger.error("Failed to load transformers AutoTokenizer: %s", e)

        self.initialized = True

    def count_tokens(self, text: str, model_type: str = "gemma") -> int:
        """Count tokens in text using the requested tokenizer type."""
        if not text:
            return 0

        model_type = model_type.lower()
        
        # Tiktoken (OpenAI)
        if "openai" in model_type or "tiktoken" in model_type or "cl100k" in model_type:
            if self.tiktoken_encoder:
                try:
                    return len(self.tiktoken_encoder.encode(text))
                except Exception as e:
                    logger.error("Error encoding with tiktoken: %s", e)
            # Fallback estimation: ~4 chars per token
            return max(1, len(text) // 4)

        # Gemma / HuggingFace AutoTokenizer
        if self.gemma_tokenizer:
            try:
                return len(self.gemma_tokenizer.encode(text))
            except Exception as e:
                logger.error("Error encoding with AutoTokenizer: %s", e)

        # General fallback: ~4 characters per token
        return max(1, len(text) // 4)
