from openai import AsyncOpenAI
import os
import base64
from typing import Optional
from .base import BaseAPIClient

class OpenAIClient(BaseAPIClient):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = AsyncOpenAI(api_key=self.api_key)

    def _get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key

    async def generate_response(self, message: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ],
                stream=True
            )

            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            return full_response
        except Exception as e:
            raise Exception(f"Error generating response from OpenAI: {str(e)}")

    async def process_image(self, image_data: bytes, prompt: str) -> str:
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error processing image with OpenAI: {str(e)}")