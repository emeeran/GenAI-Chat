import asyncio
import streamlit as st
from utils import get_api_client, stream_llm_response

async def create_database():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                chat_name TEXT,
                role TEXT,
                content TEXT
            )
        """
        )
        await db.commit()


async def add_role_column_if_not_exists():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("PRAGMA table_info(chat_history)") as cursor:
            columns = [row[1] async for row in cursor]
        if "role" not in columns:
            await db.execute("ALTER TABLE chat_history ADD COLUMN role TEXT")
            await db.commit()


async def save_chat_history_to_db(chat_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO chat_history (chat_name, role, content) VALUES (?, ?, ?)",
            [
                (chat_name, message["role"], message["content"])
                for message in st.session_state.messages
            ],
        )
        await db.commit()


async def load_chat_history_from_db(chat_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE chat_name = ? ORDER BY id",
            (chat_name,),
        ) as cursor:
            messages = [{"role": row[0], "content": row[1]} async for row in cursor]
    st.session_state.messages = messages


async def get_saved_chat_names():
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT DISTINCT chat_name FROM chat_history") as cursor:
            chat_names = [row[0] async for row in cursor]
    return chat_names


async def delete_chat(chat_name):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM chat_history WHERE chat_name = ?", (chat_name,))
        await db.commit()


async def create_content(prompt: str, content_type: str) -> str:
    full_prompt = f"Write a {content_type} based on this prompt: {prompt}"
    generated_content = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        generated_content += chunk
    return generated_content


async def summarize_text(text: str, summary_type: str) -> str:
    full_prompt = f"Please provide a {summary_type} of the following text: {text}"
    summary = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        summary += chunk
    return summary


def get_model_options(provider):
    if provider == "Groq":
        return [
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-reasoning",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "whisper-large-v3",
        ]
    elif provider == "OpenAI":
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
    elif provider == "Anthropic":
        return ["claude-v1", "claude-v1.3", "claude-instant-v1.3"]
    return []


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


async def generate_content(prompt: str, content_type: str) -> str:
    full_prompt = f"Generate {content_type} content for: {prompt}"
    generated_content = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        generated_content += chunk
    return generated_content


async def summarize_text(text: str, summary_type: str) -> str:
    full_prompt = f"Please provide a {summary_type} of the following text: {text}"
    summary = ""
    async for chunk in async_stream_llm_response(
        get_api_client(st.session_state.provider),
        st.session_state.model_params,
        [{"role": "user", "content": full_prompt}],
        st.session_state.provider,
        st.session_state.voice,
    ):
        summary += chunk
    return summary

import os
from utils import (
    setup_sidebar,
    handle_file_upload,
    get_model_options,
    get_max_token_limit,
)
from async_functions import process_chat_input

async def main():
    st.title("Multimodal Chat App")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "model": "",
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
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