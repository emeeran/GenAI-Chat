from .base import BaseAPIClient
import anthropic
import os


class AnthropicClient(BaseAPIClient):
    def __init__(self, model: str):
        self.client = anthropic.Anthropic(api_key=self._get_api_key())
        self.model = model

    def _get_api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY")

    async def process_image(self, image_path: str) -> str:
        # Implement image processing if needed, or raise NotImplementedError
        raise NotImplementedError("Image processing not implemented for Anthropic API")

    async def generate_response(self, message: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": message}],
                system="Please provide a direct, concise response.",
            )
            return response.content[0].text
        except anthropic.APIError as e:
            raise anthropic.APIError("Error generating response from Anthropic") from e


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        client = AnthropicClient(model="claude-3-5-sonnet-20241022")
        response = await client.generate_response("Hello, Claude")
        print(response)

    asyncio.run(main())
