import re
import os
import base64
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from fpdf import FPDF
from gtts import gTTS


# Utility Functions
def validate_prompt(prompt: str):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")


def sanitize_input(input_text):
    return re.sub(r"[^\w\s\-.,?!]", "", input_text).strip()


def process_uploaded_file(uploaded_file):
    # Handles the uploaded file and returns its text content
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


def perform_ocr(image_file):
    """Perform OCR on the uploaded image file."""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error performing OCR on the image: {e}")
        raise ValueError(
            "OCR processing failed. Please ensure the image is valid and readable."
        )


def process_excel_file(file):
    wb = openpyxl.load_workbook(file)
    data = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.iter_rows(values_only=True):
            data.append(" | ".join(map(str, row)))
    return "\n".join(data)


def process_ppt_file(file):
    prs = Presentation(file)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)


def text_to_speech(text: str, lang: str):
    lang_map = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
    lang_code = lang_map.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    os.remove(audio_file)
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()


def generate_openai_tts(text: str, voice: str):
    try:
        response = openai.Audio.create(
            model="tts-1",
            input=text,
            voice=voice,
        )
        audio_bytes = response.content
        st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating TTS with OpenAI: {e}")
        raise


def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(text)


def update_token_count(tokens: int):
    st.session_state.total_tokens = (
        getattr(st.session_state, "total_tokens", 0) + tokens
    )
    st.session_state.total_cost = (
        getattr(st.session_state, "total_cost", 0) + tokens * 0.0001
    )


def export_chat(format: str) -> str:
    # Check if chat history is available
    if not st.session_state.messages:
        st.warning("No chat history available to export.")
        logger.warning("Export attempted with no chat messages.")
        return None

    # Prepare chat history for export
    chat_history = "\n\n".join(
        [
            f"**{m['role'].capitalize()}:** {m['content']}"
            for m in st.session_state.messages
        ]
    ).strip()  # Remove any leading/trailing whitespace

    # Create export file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    try:
        # Write to the appropriate file format
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


# Feedback functions
def save_feedback(feedback: str):
    if "feedback.json" not in os.listdir():
        with open("feedback.json", "w") as f:
            json.dump([], f)

    with open("feedback.json", "r+") as f:
        feedback_data = json.load(f)
        feedback_data.append(feedback)
        f.seek(0)
        json.dump(feedback_data, f)