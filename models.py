# models.py

from typing import List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    token_limit: int
    provider: str

MODEL_OPTIONS = {
    "Groq": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "OpenAI": ["gpt-4o", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-opus-latest", "claude-3-5-sonnet-latest"],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
}

TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 4096,
    "mixtral-8x7b-32768": 32768,
    "claude-3-opus-latest": 200000,
    "gemini-1.5-pro": 32768,
}

class ModelRegistry:
    """Fetch model options and token limits."""
    @classmethod
    def get_models(cls, provider: str) -> List[str]:
        return MODEL_OPTIONS.get(provider, [])

    @classmethod
    def get_token_limit(cls, model_name: str) -> int:
        return TOKEN_LIMITS.get(model_name, 4096)
