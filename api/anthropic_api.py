import anthropic
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def stream_anthropic_response(client: anthropic.Anthropic, params: Dict[str, Any], messages: List[Dict[str, str]]):
    try:
        response = await client.completions.create(
            model=params["model"],
            prompt=messages[-1]["content"],
            max_tokens_to_sample=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 0.7), # Added default values
            top_p=params.get("top_p", 1.0), # Added default values
            stream=True,
        )
        async for chunk in response:
            if "completion" in chunk:
                yield chunk["completion"]
    except Exception as e:
        logger.error(f"Anthropic API Error: {e}")
        yield f"Anthropic API Error: {e}"