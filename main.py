import streamlit as st
import asyncio
import os
from functools import lru_cache
import PyPDF2
import docx
import pytesseract
from PIL import Image
import io
import pandas as pd
from io import StringIO

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
                if st.button("Rerun"):
                    st.rerun()
            with col5:
                if st.button("New"):
                    reset_current_chat()
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
            if not isinstance(st.session_state.model_params, dict):
                st.session_state.model_params = {
                    "model": model_options[0] if model_options else "",
                    "max_tokens": 1024,
                    "temperature": 1.0,

                }

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
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'pdf', 'docx', 'xlsx', 'ppt', 'jpg', 'png', 'md'])
            if uploaded_file is not None:
                try:
                    text_content = handle_file_upload(uploaded_file)
                    st.session_state.file_content = text_content
                    st.success(f"File '{uploaded_file.name}' loaded successfully! You can now chat about its contents.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # with st.expander("Summarize"):
        #     st.session_state.show_summarization = st.checkbox(
        #         "Enable Summarization", value=False
        #     )
        #     if st.session_state.show_summarization:
        #         st.session_state.summarization_type = st.selectbox(
        #             "Summarization Type:",
        #             [
        #                 "Main Takeaways",
        #                 "Main points bulleted",
        #                 "Concise Summary",
        #                 "Executive Summary",
        #             ],
        #         )

        # with st.expander("Content Generation"):
        #     st.session_state.content_creation_mode = st.checkbox(
        #         "Enable Content Creation Mode", value=False
        #     )
        #     if st.session_state.content_creation_mode:
        #         st.session_state.content_type = st.selectbox(
        #             "Select Content Type:", list(CONTENT_TYPES.keys())
        #         )

        with st.expander("Export"):
            export_format = st.selectbox(
                "Export Format", ["md", "pdf", "txt", "docx", "json"]
            )
            if st.button("Export Chat"):
                filename = export_chat(export_format)
                if (
                    filename
                ):  # Check if filename is valid before showing download button
                    st.success("Chat exported successfully!")
                    # Provide a download button for the user
                    with open(filename, "rb") as f:
                        st.download_button(
                            label="Download Chat",
                            data=f,
                            file_name=os.path.basename(filename),
                            mime="application/octet-stream",
                        )

        # st.session_state.color_scheme = st.selectbox("Color Scheme", ["Light", "Dark"])
        # if st.session_state.color_scheme == "Dark":
        #     st.markdown(
        #         """
        #         <style>
        #         .stApp {
        #             background-color: #1E1E1E;
        #             color: #FFFFFF;
        #         }
        #         </style>
        #         """,
        #         unsafe_allow_html=True,
        #     )


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


if __name__ == "__main__":
    st.set_page_config(page_title="Multimodal Chat App", page_icon="💬")
    asyncio.run(main())