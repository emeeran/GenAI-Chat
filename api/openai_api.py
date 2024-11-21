import openai
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def stream_openai_response(api_key, params, messages):
    try:
        async with openai.AsyncOpenAI(api_key=api_key) as client:
            response = await client.chat.completions.create(
                model=params["model"],
                messages=messages,
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7), # Added default values
                top_p=params.get("top_p", 1.0), # Added default values
                stream=True,
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                yield content
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        yield f"OpenAI API Error: {e}"
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        yield f"OpenAI API Error: {e}"