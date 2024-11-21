import streamlit as st
import asyncio
import os
from functools import lru_cache

# Absolute imports - specify the full path from the project root
from config import *
from utils import *
from database import *
from api.openai_api import stream_openai_response
from api.groq_api import stream_groq_response, GroqAPIError
from api.anthropic_api import stream_anthropic_response
from persona import PERSONAS
from content_type import CONTENT_TYPES
from io import StringIO
import pandas as pd

@lru_cache(maxsize=None)
def get_api_client(provider):
    if provider == "Groq":
        return GROQ_API_KEY
    elif provider == "OpenAI":
        return OPENAI_API_KEY
    elif provider == "Anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return None


async def stream_llm_response(client, params, messages, provider):
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


async def process_chat_input(prompt):
    messages = st.session_state.messages + [{"role": "user", "content": prompt}]
    full_response = ""

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        client = get_api_client(st.session_state.provider)
        if client is None:
            message_placeholder.error("API Key not set for selected provider!")
            return

        async for chunk in stream_llm_response(client, st.session_state.model_params, messages, st.session_state.provider):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


def handle_file_upload():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


def setup_sidebar():
    st.sidebar.title("Settings")
    st.session_state.provider = st.sidebar.selectbox("Provider", PROVIDER_OPTIONS)

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

    selected_model = st.session_state.get("model", model_options.get(st.session_state.provider, [""])[0])

    st.session_state.model_params = {
        "model": st.sidebar.selectbox("Model", model_options.get(st.session_state.provider, [""]), key="model"),
        "max_tokens": st.sidebar.number_input("Max Tokens", 1, 4000, 1024),
        "temperature": st.sidebar.slider("Temperature", 0.0, 2.0, 0.7),
        "top_p": st.sidebar.slider("Top p", 0.0, 1.0, 1.0),
    }

    st.session_state.enable_audio = st.sidebar.checkbox("Enable Audio Response")
    handle_file_upload()


async def main():
    st.title("Multimodal Chat App")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    setup_sidebar()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Enter your message:")
    if prompt:
        await process_chat_input(prompt)


if __name__ == "__main__":
    st.set_page_config(page_title="Multimodal Chat App", page_icon="💬")
    asyncio.run(main())