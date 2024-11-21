from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os

class BaseAPIClient(ABC):
    """Base class for API clients."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = self._get_api_key()

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key for the service."""
        pass

    @abstractmethod
    async def generate_response(self, message: str) -> str:
        """Generate response from the API."""
        pass

    @abstractmethod
    async def process_image(self, image_data: bytes, prompt: str) -> str:
        """Process image and generate response."""
        pass

def get_api_client(provider: str, model: str) -> BaseAPIClient:
    """Factory function to get the appropriate API client."""
    if provider.lower() == "anthropic":
        from app.api.anthropic_client import AnthropicClient
        return AnthropicClient(model)
    elif provider.lower() == "openai":
        from app.api.openai_client import OpenAIClient
        return OpenAIClient(model)
    elif provider.lower() == "groq":
        from app.api.groq_client import GroqClient
        return GroqClient(model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")