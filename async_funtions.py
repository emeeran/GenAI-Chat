import asyncio
import streamlit as st
from utils import get_api_client, stream_llm_response, setup_sidebar, handle_file_upload, get_model_options, get_max_token_limit

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

async def process_chat_input(prompt: str, client: Any) -> None:
    try:
        validate_prompt(prompt)

        if (
            st.session_state.file_content
            and not st.session_state.is_file_response_handled
        ):
            prompt = f"Based on the uploaded file content, {prompt}\n\nFile content: {st.session_state.file_content[:MAX_FILE_CONTENT_LENGTH]}{TRUNCATION_ELLIPSIS}"
            st.session_state.is_file_response_handled = True

        persona_content = (
            st.session_state.custom_persona
            if st.session_state.persona == "Custom"
            else PERSONAS[st.session_state.persona]
        )

        messages = [
            {"role": "system", "content": persona_content},
            *st.session_state.messages[-MAX_CHAT_HISTORY_LENGTH:],
            {"role": "user", "content": prompt},
        ]
        with st.chat_message("user"):
            st.markdown(prompt)

        full_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            async for chunk in async_stream_llm_response(
                client,
                st.session_state.model_params,
                messages,
                st.session_state.provider,
                st.session_state.voice,
            ):
                if chunk.startswith("API Error:"):
                    message_placeholder.error(chunk)
                    return
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_response},
            ]
        )

        logger.info(f"Current messages in session state: {st.session_state.messages}")

        feedback_container = st.container()
        with feedback_container:
            st.markdown("### How was the response?")
            feedback = st.radio(
                "Rate this response:", options=["👍 Good", "👎 Poor"], key="feedback"
            )
            if st.button("Submit Feedback"):
                save_feedback({"feedback": feedback, "response": full_response})
                st.success("Thanks for your feedback!")

        if st.session_state.enable_audio and full_response.strip():
            if st.session_state.provider == "OpenAI":
                generate_openai_tts(full_response, st.session_state.voice)
            else:
                text_to_speech(full_response, st.session_state.language)
            st.audio(
                f"data:audio/mp3;base64,{st.session_state.audio_base64}",
                format="audio/mp3",
            )

        update_token_count(len(full_response.split()))

        if st.session_state.content_creation_mode:
            content_type = CONTENT_TYPES[st.session_state.content_type]
            generated_content = await generate_content(prompt, content_type)
            with st.chat_message("assistant"):
                st.markdown(
                    f"## Generated {st.session_state.content_type}:\n\n{generated_content}"
                )

        if st.session_state.show_summarization:
            text_to_summarize = st.session_state.file_content or prompt
            summary = await summarize_text(
                text_to_summarize, st.session_state.summarization_type
            )
            with st.chat_message("assistant"):
                st.markdown(
                    f"## Summary ({st.session_state.summarization_type}):\n\n{summary}"
                )

    except ValueError as ve:
        st.error(f"Invalid input: {str(ve)}")
    except openai.APIError as e:
        st.error(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in process_chat_input: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred. Please try again later.")

async def export_chat(chat_name: str, format: str = "txt") -> str:
    chat_content = ""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT role, content FROM chat_history WHERE chat_name = ? ORDER BY id",
            (chat_name,),
        ) as cursor:
            async for row in cursor:
                chat_content += f"{row[0]}: {row[1]}\n\n"
    return chat_content

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
