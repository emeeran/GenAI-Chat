import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

MAX_CHAT_HISTORY_LENGTH = 50
DB_PATH = "chat_history.db"
MAX_FILE_CONTENT_LENGTH = 4000
TRUNCATION_ELLIPSIS = "..."
CHUNK_SIZE = 2000

VOICE_OPTIONS = {
    "OpenAI": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "gTTS": ["en", "ta", "hi"],
}

PROVIDER_OPTIONS = ["Groq", "OpenAI", "Anthropic"]