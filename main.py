import streamlit as st
import asyncio
import os
from functools import lru_cache
import pandas as pd

# Import utility functions from utils.py
from utils import (
    perform_ocr,
    process_excel_file,
    process_ppt_file,
    get_max_token_limit,
    get_model_options,
    get_api_client,
    stream_llm_response,
    process_chat_input,
    handle_file_upload,
    export_chat,
    process_file_content,
    update_file_context,
    reset_current_chat,  # Add this import
    save_session_state,
    load_session_state,
)

# Absolute imports - specify the full path from the project root
from config import *
from database import *
from api.openai_api import stream_openai_response
from api.groq_api import stream_groq_response, GroqAPIError
from api.anthropic_api import stream_anthropic_response
from persona import PERSONAS
from content_type import CONTENT_TYPES

def setup_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align: center;color: #6ca395;'>Select Provider</h2>",
            unsafe_allow_html=True,
        )

        available_providers = [
            p for p in PROVIDER_OPTIONS if os.getenv(f"{p.upper()}_API_KEY")
        ]

        if not available_providers:
            st.error(
                "No API keys are set. Please set at least one API key in your .env file."
            )
            st.stop()

        selected_provider = st.selectbox(
            "Provider",
            available_providers,
            label_visibility="collapsed",
            format_func=lambda x: "Select Provider" if x == "" else x,
        )
        st.session_state.provider = selected_provider

        st.markdown(
            "<h2 style='text-align: center;'>Settings 🛠️ </h2> ", unsafe_allow_html=True
        )

        with st.expander("Chat Settings", expanded=False):
            saved_chats = st.session_state.get("saved_chats", [])
            selected_chat = st.selectbox(
                "Load Chat History", options=[""] + saved_chats
            )
            if selected_chat:
                st.session_state.load_chat = selected_chat

            col4, col5, col6 = st.columns(3)
            with col4:
                if st.button("🔄 Rerun"):
                    rerun()  # Use experimental_rerun instead of rerun
            with col5:
                if st.button("New"):
                    reset_current_chat()
                    st.rerun()
            with col6:
                if st.button("Delete"):
                    if selected_chat:
                        st.session_state.delete_chat = selected_chat
                        st.rerun()

            col1, col2 = st.columns([2, 1])
            with col1:
                chat_name_input = st.text_input(
                    "Enter a name for this chat:",
                    max_chars=50,
                    label_visibility="collapsed",
                    placeholder="Chat Name",
                    help="Type a name for your chat",
                )
            with col2:
                if st.button("Save"):
                    if chat_name_input:
                        st.session_state.save_chat = chat_name_input
                        st.rerun()

        with st.expander("Model"):
            model_options = get_model_options(st.session_state.provider)
            st.session_state.model_params = st.session_state.get("model_params", {
                "model": model_options[0] if model_options else "",
                "max_tokens": 1024,
                "temperature": 1.0,
            })

            st.session_state.model_params["model"] = st.selectbox(
                "Choose Model:",
                options=model_options,
                index=(
                    model_options.index(st.session_state.model_params["model"])
                    if st.session_state.model_params["model"] in model_options
                    else 0
                ),
            )

            max_token_limit = get_max_token_limit(
                st.session_state.model_params["model"]
            )
            st.session_state.model_params["max_tokens"] = st.slider(
                "Max Tokens:",
                min_value=1,
                max_value=max_token_limit,
                value=min(st.session_state.model_params["max_tokens"], max_token_limit),
                step=1,
            )
            st.session_state.model_params["temperature"] = st.slider(
                "Temperature:",
                0.0,
                2.0,
                st.session_state.model_params["temperature"],
                0.1,
            )

        with st.expander("Persona"):
            persona_options = list(PERSONAS.keys()) + ["Custom"]
            st.session_state.persona = st.selectbox(
                "Select Persona:",
                options=persona_options,
                index=persona_options.index("Default"),
            )

            if st.session_state.persona == "Custom":
                custom_persona = st.text_area(
                    "Enter Custom Persona Description:",
                    value=st.session_state.get("custom_persona", ""),
                    height=100,
                    help="Describe the persona you want the AI to adopt.",
                )
                st.session_state.custom_persona = custom_persona
            else:
                st.text_area(
                    "Persona Description:",
                    value=PERSONAS[st.session_state.persona],
                    height=100,
                    disabled=True,
                )

        with st.expander("Audio & Language"):
            st.session_state.enable_audio = st.checkbox(
                "Enable Audio Response", value=False
            )
            language_options = ["English", "Tamil", "Hindi"]
            st.session_state.language = st.selectbox(
                "Select Language:", language_options
            )
            voice_options = (
                VOICE_OPTIONS["OpenAI"]
                if st.session_state.provider == "OpenAI"
                else VOICE_OPTIONS["gTTS"]
            )
            st.session_state.voice = st.selectbox(
                f"Select {'Voice' if st.session_state.provider == 'OpenAI' else 'Language Code'}:",
                voice_options,
            )

        with st.expander("File Upload"):
            st.session_state.uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv', 'pdf', 'docx', 'xlsx', 'ppt', 'jpg', 'png', 'md']
            )

            if st.session_state.uploaded_file is not None:
                try:
                    text_content = handle_file_upload(st.session_state.uploaded_file)
                    st.session_state.file_content = text_content
                    st.success(f"File '{st.session_state.uploaded_file.name}' loaded successfully!")

                    # File operations
                    st.divider()
                    col1, col2 = st.columns([3,1])

                    with col1:
                        operation = st.selectbox(
                            "Select operation",
                            ["Chat", "Summarize", "Bullets", "Analyze"],
                            key="file_operation"
                        )

                    with col2:
                        if st.button("Go🏃", type="primary"):
                            if operation == "Chat":
                                update_file_context()
                                st.info("File loaded! You can now chat about its contents.", icon="💬")
                            else:
                                st.session_state.processing_operation = operation.lower()
                                st.session_state.processing_content = text_content

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        with st.expander("Export"):
            export_format = st.selectbox(
                "Export Format", ["md", "pdf", "txt", "docx", "json"]
            )
            if st.button("Export Chat"):
                filename = export_chat(export_format)
                if filename:
                    st.success("Chat exported successfully!")
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="Download Chat",
                            data=f,
                            file_name=os.path.basename(filename),
                            mime="application/octet-stream",
                        )

def setup_file_upload():
    """Setup file upload and processing section"""
    # Remove outer expander - already inside sidebar expander
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'csv', 'pdf', 'docx', 'xlsx', 'ppt', 'jpg', 'png', 'md']
    )

    if uploaded_file is not None:
        try:
            text_content = handle_file_upload(uploaded_file)
            st.session_state.file_content = text_content
            st.success(f"File '{uploaded_file.name}' loaded successfully!")

            # File operations
            st.divider()
            col1, col2 = st.columns([3,1])

            with col1:
                operation = st.selectbox(
                    "Select operation",
                    ["Chat", "Summarize", "Bullets", "Analyze"],
                    key="file_operation"
                )

            with col2:
                if st.button("Process", type="primary"):
                    if operation == "Chat":
                        update_file_context()
                        st.info("File loaded! You can now chat about its contents.", icon="💬")
                    else:
                        st.session_state.processing_operation = operation.lower()
                        st.session_state.processing_content = text_content

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    return uploaded_file

async def main():
    st.markdown(
            '<h1 style="text-align: center; color: #6ca395;">GenAI-Chat 💬</h1>',
            unsafe_allow_html=True,
        )
    st.markdown(
            '<p style="text-align: center; color : #74a6d4">Experience the power of AI!</p>',
            unsafe_allow_html=True,
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "model": "",
            "max_tokens": 1024,
            "temperature": 1.0,
        }

    setup_sidebar()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Enter your message:")
    if prompt:
        await process_chat_input(prompt)

    if hasattr(st.session_state, 'processing_operation'):
        process_file_content(
            st.session_state.processing_content,
            st.session_state.processing_operation
        )
        # Clear processing state
        del st.session_state.processing_operation
        del st.session_state.processing_content


if __name__ == "__main__":
    st.set_page_config(page_title="GenAI-Chat", page_icon="💬")
    asyncio.run(main())
