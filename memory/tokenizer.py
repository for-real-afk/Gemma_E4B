import logging

logger = logging.getLogger(__name__)


class TokenCalculator:
    """
    Token counting for cost estimation.

    Primary counts come from Ollama's native prompt_eval_count / eval_count
    fields — those are exact and require no local tokenizer.

    This class is the fallback only (used when Ollama doesn't return counts,
    or for attachment text sizing). It loads only tiktoken (instant, no
    network) and skips HuggingFace downloads entirely — those took 15+
    HTTP requests on every server restart and are unnecessary now that
    Ollama provides native counts.

    Fallback chain:
      GPT-4 model  →  tiktoken cl100k_base  →  char estimate (len // 4)
      Gemma model  →  char estimate (len // 4)
    """

    def __init__(self):
        self.tiktoken_encoder = None
        self.initialized      = False

    def initialize(self) -> None:
        """Load tiktoken only. Fast, no network calls."""
        try:
            import tiktoken
            self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("tiktoken encoder 'cl100k_base' loaded.")
        except Exception as e:
            logger.warning("tiktoken unavailable, will use char estimate: %s", e)
        self.initialized = True
        logger.info("TokenCalculator ready (Ollama native counts are primary).")

    @staticmethod
    def route_tokenizer(model: str) -> str:
        if model.lower().strip() in ("gpt4", "gpt-4", "gpt"):
            return "openai"
        return "gemma"

    def count_messages_tokens(self, messages: list, model: str) -> int:
        tokenizer_type = self.route_tokenizer(model)
        return sum(
            self.count_tokens(msg.get("content", ""), tokenizer_type)
            for msg in messages
            if msg.get("content")
        )

    def count_tokens(self, text: str, model_type: str = "gemma") -> int:
        if not text:
            return 0

        if "openai" in model_type.lower():
            if self.tiktoken_encoder:
                try:
                    return len(self.tiktoken_encoder.encode(text))
                except Exception:
                    pass

        # Gemma or any other model — char estimate
        return max(1, len(text) // 4)
