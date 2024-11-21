import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from groq import AsyncGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqAPIError(Exception):
    """Custom exception for Groq API errors."""
    pass

async def stream_groq_response(
    api_key: str,
    params: Dict[str, Any],
    messages: list[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """Streams responses from the Groq API using the official SDK."""
    async with AsyncGroq(api_key=api_key) as client:  # Use the AsyncGroq client
        try:
            stream = await client.chat.completions.create(
                model=params["model"],
                messages=messages,
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                stream=True,
            )
            async for chunk in stream:
                if content := chunk.choices[0].delta.content: #More efficient content extraction
                    yield content

        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            raise GroqAPIError(f"Groq API Error: {str(e)}") from e