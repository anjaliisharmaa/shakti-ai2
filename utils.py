# helper: text extraction + translation
# not using for now






from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from googletrans import GoogleTranslator

def extract_text_from_file(file):
    file_type = file.type
    if "pdf" in file_type:
        return extract_text_from_pdf(file)
    elif "image" in file_type:
        return extract_text_from_image(file)
    return "Unsupported file type."

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def translate_text(text, target_lang="hi"):
    if not text.strip():
        return ""
    translator = GoogleTranslator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text
