# Standard library imports
import os
import re
import logging
import json
import base64
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Any
import time
from pathlib import Path

# Third party imports
import streamlit as st
import requests
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from fpdf import FPDF
from gtts import gTTS
import PyPDF2
import docx
import openpyxl
from pptx import Presentation
import openai
from transformers import AutoTokenizer

# Local imports
from api.groq_api import stream_groq_response, GroqAPIError
from api.openai_api import stream_openai_response
from api.anthropic_api import stream_anthropic_response

# Define file type handlers
file_handlers = {
    "application/pdf": lambda f: " ".join(
        page.extract_text() for page in PyPDF2.PdfReader(f).pages
    ),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(
        paragraph.text for paragraph in docx.Document(f).paragraphs
    ),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": lambda f: process_excel_file(f),
    "application/vnd.ms-powerpoint": lambda f: process_ppt_file(f),
    "text/plain": lambda f: f.getvalue().decode("utf-8"),
    "text/markdown": lambda f: f.getvalue().decode("utf-8"),
    "image/jpeg": lambda f: perform_ocr(f),
    "image/png": lambda f: perform_ocr(f),
    "text/csv": lambda f: pd.read_csv(f).to_string(),
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Utility Functions
def validate_prompt(prompt: str):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

def sanitize_input(input_text: str) -> str:
    return re.sub(r"[^\w\s\-.,?!]", "", input_text).strip()

def perform_ocr(file_obj):
    try:
        image = Image.open(file_obj)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No text found in the image"
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        return f"Error processing image: {str(e)}"

def process_excel_file(file_obj):
    wb = openpyxl.load_workbook(file_obj)
    data = [
        " | ".join(map(str, row))
        for sheet in wb.sheetnames
        for row in wb[sheet].iter_rows(values_only=True)
    ]
    return "\n".join(data)

def process_ppt_file(file_obj):
    try:
        prs = Presentation(file_obj)
        text = [
            shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")
        ]
        logger.info("Successfully processed PowerPoint file")
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error processing PowerPoint file: {str(e)}", exc_info=True)
        return f"Error processing PowerPoint file: {str(e)}"

def get_max_token_limit(model: str) -> int:
    token_limits = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        "claude-3-opus-latest": 200000,
        "claude-3-sonnet-latest": 200000,
        "claude-3-haiku-20240307": 200000,
        "llama-3.1-70b-versatile": 8192,
        "mixtral-8x7b-32768": 32768
    }
    return token_limits.get(model, 4096)

def get_model_options(provider: str):
    model_options = {
        "Groq": [
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-reasoning",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "whisper-large-v3",
        ],
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "Anthropic": [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-latest",
        ]
    }
    return model_options.get(provider, [])

@lru_cache(maxsize=None)
def get_api_client(provider: str):
    if provider == "Groq":
        return GROQ_API_KEY
    elif provider == "OpenAI":
        return OPENAI_API_KEY
    elif provider == "Anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return None

async def stream_llm_response(client, params, messages, provider: str):
    try:
        if provider == "Groq":
            async for chunk in stream_groq_response(client, params, messages):
                yield chunk
        elif provider == "OpenAI":
            async for chunk in stream_openai_response(client, params, messages):
                yield chunk
        elif provider == "Anthropic":
            async for chunk in stream_anthropic_response(client, params, messages):
                yield chunk
        else:
            yield "Error: Unsupported provider"
    except GroqAPIError as e:
        yield f"Groq API Error: {e}"
    except Exception as e:
        yield f"API Error: {str(e)}"

def display_chat_message(message: Dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
            logger.info(f"User message: {message['content'][:100]}...")
        else:
            st.markdown(message["content"])

async def process_chat_input(prompt: str) -> None:
    user_message = {"role": "user", "content": prompt}
    display_chat_message(user_message)
    st.session_state.messages.append(user_message)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        client = get_api_client(st.session_state.provider)
        if client is None:
            message_placeholder.error("API Key not set for selected provider!")
            return

        full_response = ""
        async for chunk in stream_llm_response(
            client,
            st.session_state.model_params,
            st.session_state.messages,
            st.session_state.provider
        ):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # Handle audio response if enabled
        if st.session_state.enable_audio:
            try:
                if st.session_state.provider == "OpenAI":
                    generate_openai_tts(full_response, st.session_state.voice)
                else:
                    text_to_speech(full_response, st.session_state.language)

                if hasattr(st.session_state, 'audio_base64'):
                    st.audio(
                        data=base64.b64decode(st.session_state.audio_base64),
                        format='audio/mp3'
                    )
            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")
                logger.error(f"Audio generation error: {str(e)}", exc_info=True)

def handle_file_upload(uploaded_file):
    """Handle file upload with token limit awareness"""
    try:
        content = None
        for file_type, handler in file_handlers.items():
            if uploaded_file.type.startswith(file_type):
                content = handler(uploaded_file)
                break

        if content is None:
            raise ValueError("Unsupported file type")

        # Check token count
        token_count = len(tokenizer.encode(content))
        if token_count > 5000:
            logger.warning(f"Large content detected ({token_count} tokens). Content will be processed in chunks.")

        return content

    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        raise

def text_to_speech(text: str, lang: str):
    try:
        lang_map = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
        lang_code = lang_map.get(lang, "en")
        tts = gTTS(text=text, lang=lang_code)
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        os.remove(audio_file)
        st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.error(f"gTTS error: {str(e)}")
        raise

def generate_openai_tts(text: str, voice: str):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        audio_bytes = response.content
        st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        raise

def sync_process_chat(prompt: str) -> None:
    """Synchronous wrapper for async chat processing"""
    import nest_asyncio
    import asyncio

    # Apply patch for nested event loops
    nest_asyncio.apply()

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run async function
    loop.run_until_complete(process_chat_input(prompt))

def chunk_content(text: str, max_tokens: int = 5000) -> List[str]:
    """Split content into chunks that fit within token limits"""
    chunks = []
    current_chunk = ""
    current_tokens = 0

    sentences = text.split('. ')
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '. '
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def process_file_content(content: str, operation: str) -> None:
    """Process file content based on selected operation"""
    try:
        # Map operations to prompts
        operation_prompts = {
            "summarize": "Please provide a concise summary of this content:\n\n",
            "bullets": "Extract and list the main points as bullet points from this content:\n\n",
            "bullet_points": "Extract and list the main points as bullet points from this content:\n\n",
            "analyze": "Analyze the key themes and insights from this content:\n\n"
        }

        # Get prompt template or raise error for invalid operation
        if operation not in operation_prompts:
            raise ValueError(f"Invalid operation: {operation}")

        prompt = operation_prompts[operation] + content

        # Process chunks if content is large
        chunks = chunk_content(content)
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(1)  # Rate limiting
            chunk_prompt = f"{operation_prompts[operation]} (Part {i+1}/{len(chunks)}):\n\n{chunk}"
            sync_process_chat(chunk_prompt)

        if len(chunks) > 1:
            sync_process_chat("Please provide a final combined summary of all the parts discussed above.")

    except Exception as e:
        logger.error(f"Error processing file content: {str(e)}")
        raise

def update_file_context():
    """Update chat context with loaded file content"""
    if hasattr(st.session_state, 'file_content'):
        system_message = {
            "role": "system",
            "content": f"Context from uploaded file:\n{st.session_state.file_content[:1000]}..."
        }
        if not st.session_state.messages:
            st.session_state.messages = [system_message]
        else:
            st.session_state.messages.insert(0, system_message)

def export_chat(export_format: str) -> Optional[str]:
    """Export chat history in the specified format"""
    try:
        if not st.session_state.messages:
            logger.warning("No chat history to export")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_export_{timestamp}.{export_format}"

        if export_format == "json":
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, indent=2, ensure_ascii=False)

        elif export_format == "txt":
            with open(filename, "w", encoding="utf-8") as f:
                for msg in st.session_state.messages:
                    f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")

        elif export_format == "md":
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# Chat Export\n\n")
                for msg in st.session_state.messages:
                    f.write(f"### {msg['role'].title()}\n{msg['content']}\n\n")

        elif export_format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for msg in st.session_state.messages:
                pdf.cell(200, 10, txt=f"{msg['role'].upper()}:", ln=True)
                pdf.multi_cell(200, 10, txt=msg['content'])
                pdf.ln()
            pdf.output(filename)

        elif export_format == "docx":
            doc = docx.Document()
            doc.add_heading('Chat Export', 0)
            for msg in st.session_state.messages:
                doc.add_heading(msg['role'].title(), level=2)
                doc.add_paragraph(msg['content'])
                doc.add_paragraph()
            doc.save(filename)

        logger.info(f"Chat exported to {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error exporting chat: {str(e)}", exc_info=True)
        return None

def reset_current_chat():
    """Reset chat history and session state"""
    # Clear messages
    st.session_state.messages = []

    # Reset model parameters
    st.session_state.model_params = {
        "model": "",
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
    }

    # Clear file related states
    keys_to_clear = [
        'file_content',
        'uploaded_file',
        'processing_operation',
        'processing_content'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    logger.info("Chat session reset")

def save_chat(chat_name: str) -> bool:
    """Save current chat session"""
    try:
        if not st.session_state.messages:
            logger.warning("No chat history to save")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chat_name}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, indent=2, ensure_ascii=False)

        logger.info(f"Chat saved to {filename}")
        return True

    except Exception as e:
        logger.error(f"Error saving chat: {str(e)}", exc_info=True)
        return False

def save_session_state(chat_name: str, session_dir: str = "sessions") -> bool:
    """Save chat history and settings with a given name"""
    try:
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chat_name}_{timestamp}.json"
        filepath = Path(session_dir) / filename

        # Save messages and settings
        chat_state = {
            "messages": st.session_state.messages,
            "settings": {
                "model": st.session_state.model_params["model"],
                "temperature": st.session_state.model_params["temperature"],
                "persona": st.session_state.persona,
                "custom_persona": st.session_state.get("custom_persona", "")
            },
            "timestamp": timestamp
        }

        with open(filepath, 'w') as f:
            json.dump(chat_state, f, indent=2)

        # Update saved chats list
        if "saved_chats" not in st.session_state:
            st.session_state.saved_chats = []
        st.session_state.saved_chats.append(filename)

        return True
    except Exception as e:
        logging.error(f"Failed to save chat: {e}")
        return False

def load_session_state(filepath: str) -> Optional[Dict[str, Any]]:
    """Load saved chat history"""
    try:
        with open(filepath, 'r') as f:
            state = json.load(f)
        return state
    except Exception as e:
        logging.error(f"Failed to load chat: {e}")
        return None