import anthropic
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def stream_anthropic_response(client: anthropic.Anthropic, params: Dict[str, Any], messages: List[Dict[str, str]]):
    try:
        response = client.messages.create(
            model=params["model"],
            messages=messages,
            max_tokens=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 1.0),
            stream=True,
        )
        for chunk in response:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
    except Exception as e:
        logger.error(f"Anthropic API Error: {e}")
        yield f"Anthropic API Error: {e}"