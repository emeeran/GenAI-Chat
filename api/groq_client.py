import requests
import os
from typing import Optional
from .base import BaseAPIClient

class GroqClient(BaseAPIClient):
    def __init__(self, model: str):
        super().__init__(model)
        self.base_url = "https://api.groq.com/openai/v1"

    def _get_api_key(self) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return api_key

    async def generate_response(self, message: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Error generating response from Groq: {str(e)}")

    async def process_image(self, image_data: bytes, prompt: str) -> str:
        # Implement when Groq adds image support
        raise NotImplementedError("Image processing not yet supported for Groq")