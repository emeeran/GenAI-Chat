# Standard library imports
import os
import re
import logging
import json
import base64
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

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

# Local imports
from api.groq_api import stream_groq_response, GroqAPIError
from api.openai_api import stream_openai_response
from api.anthropic_api import stream_anthropic_response

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
    file_handlers = {
        "application/pdf": lambda f: " ".join(
            page.extract_text() for page in PyPDF2.PdfReader(f).pages
        ),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": lambda f: " ".join(
            paragraph.text for paragraph in docx.Document(f).paragraphs
        ),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.document": lambda f: process_excel_file(
            f
        ),
        "application/vnd.ms-powerpoint": lambda f: process_ppt_file(f),
        "text/plain": lambda f: f.getvalue().decode("utf-8"),
        "text/markdown": lambda f: f.getvalue().decode("utf-8"),
        "image/jpeg": lambda f: perform_ocr(f),
        "image/png": lambda f: perform_ocr(f),
    }

    for file_type, handler in file_handlers.items():
        if uploaded_file.type.startswith(file_type):
            return handler(uploaded_file)

    raise ValueError("Unsupported file type")

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

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)

def save_feedback(feedback: str):
    if "feedback.json" not in os.listdir():
        with open("feedback.json", "w") as f:
            json.dump([], f)

    with open("feedback.json", "r+") as f:
        feedback_data = json.load(f)
        feedback_data.append(feedback)
        f.seek(0)
        json.dump(feedback_data, f)

def reset_current_chat():
    st.session_state.messages = []
    st.session_state.is_file_response_handled = False

def is_connected():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def save_chat_history_locally():
    chat_data = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    with open("local_chat_history.json", "w") as f:
        json.dump(chat_data, f)

def load_chat_history_locally():
    if os.path.exists("local_chat_history.json"):
        with open("local_chat_history.json", "r") as f:
            return json.load(f)
    return []

def update_token_count(tokens: int):
    st.session_state.total_tokens = (
        getattr(st.session_state, "total_tokens", 0) + tokens
    )
    st.session_state.total_cost = (
        getattr(st.session_state, "total_cost", 0) + tokens * 0.0001
    )

def export_chat(format: str) -> str:
    if not st.session_state.messages:
        st.warning("No chat history available to export.")
        logger.warning("Export attempted with no chat messages.")
        return None

    chat_history = "\n\n".join(
        [
            f"**{m['role'].capitalize()}:** {m['content']}"
            for m in st.session_state.messages
        ]
    ).strip()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    try:
        if format == "md":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chat_history)
        elif format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, chat_history)
            pdf.output(filename)
        elif format == "txt":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chat_history)
        elif format == "docx":
            from docx import Document
            doc = Document()
            doc.add_paragraph(chat_history)
            doc.save(filename)
        elif format == "json":
            chat_data = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ]
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, indent=4)

        logger.info(f"Chat exported successfully to {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error exporting chat: {e}", exc_info=True)
        st.error(f"An error occurred while exporting the chat: {e}")
        return None
