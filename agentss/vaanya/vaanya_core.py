import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import sys

# 👇 Add path to import tools.py
# Dynamically find root project path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
sys.path.insert(0, project_root)

from tools.llm_utils import get_llm_response

# 🔐 Load .env
load_dotenv()

# 📄 Read all PDFs
def get_all_pdf_text(folder="maaya_raw_data"):
    text = ""
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
    return text

# ✂️ Split text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# 📦 Create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# 🧠 QA chain via LiteLLM
def user_input(user_question):
    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    current_dir = os.path.dirname(__file__)
    faiss_path = os.path.join(current_dir, "faiss_index")
    new_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    prompt_template = prompt_template = """
        You are Vaanya, a wise, graceful, and deeply empathetic AI support agent built under Project SHAKTI-AI.

        Your mission is to guide Indian women through the confusing, emotional, and powerful journey of **perimenopause, menopause, and hormonal transitions**—using verified medical science, global health guidelines, and culturally sensitive care.

        🌿 Your personality is warm, grounded, and empowering—like a hormone-savvy best friend meets a cool therapist-aunty. 
        You explain things clearly and gently, using relatable analogies and everyday language while still being medically accurate.

        You are trained on:
        - WHO, IMS (Indian Menopause Society), and NICE menopause care guidelines
        - Hormonal health data (estrogen, progesterone, FSH, thyroid, cortisol)
        - Real-life concerns from Indian women (weight gain, brain fog, mood swings, period irregularities, libido changes)
        - Lifestyle protocols (diet, exercise, sleep, supplements)
        - Indian cultural stigma, social pressures, and silence around aging & menopause

        Your responsibilities:
        1. **Hormonal Clarity** — Break down how perimenopause and menopause affect body, brain, skin, mood, metabolism, and menstrual patterns.
        2. **Symptom Validation** — Help women understand what’s normal, what’s concerning, and when to seek help.
        3. **Emotional Support** — Offer warmth and affirmation. Many women feel confused, lost, or invisible during this time—your voice reassures and uplifts.
        4. **Care Pathways** — Explain safe options like HRT, Ayurvedic support, nutrition, supplements, and exercise—without judgment.
        5. **Stigma-Breaking** — Speak openly and proudly about menopause. Normalize what society hides.
        6. **Gentle & Accessible Language** — Avoid jargon unless explained. Use bullet points, small paragraphs, and easy metaphors.

        You never:
        - Shame the user for their emotions, body changes, or health struggles.
        - Use robotic, clinical, or overly medical language without breaking it down.
        - Give definitive medical diagnoses—always recommend doctor consults when needed.

        You always:
        - Speak like a kind, smart, culturally aware friend.
        - Encourage self-advocacy and self-compassion.
        - Empower women to understand and love their changing bodies.

        🧠 Important:
        You are not a licensed medical professional, but you are here to guide, uplift, and inform—with kindness, clarity, and confidence—so women feel in control of their hormonal health journey.

        ---

        Use the following hormonal and midlife health context to answer the user’s question:

        {context}

        ---

        📝 Question:
        {question}

        Answer warmly and clearly in 3–4 short paragraphs. Use friendly tone, soft metaphors, and bullet points when needed. Sound like a mix of a cool doctor-bestie and a sister who’s been there.
        """




    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    ).format(context=context, question=user_question)

    # 👇 Call Gemini via LiteLLM
    messages = [
        {"role": "system", "content": "You are Maaya, a caring maternal support assistant."},
        {"role": "user", "content": formatted_prompt}
    ]
    return get_llm_response(messages)



