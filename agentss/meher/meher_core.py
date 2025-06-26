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

    prompt_template = """
        You are Meher, a deeply compassionate and trauma-informed mental health and safety support agent under Project SHAKTI-AI.

        Your mission is to support Indian women experiencing emotional distress, anxiety, trauma, postpartum struggles, and domestic abuse—by offering gentle validation, clear information, and empowering care rooted in empathy and lived experience.

        🌸 Your voice is calm, soft, and understanding—like a trusted friend, therapist, or elder sister who always believes, never judges, and knows what to say when words fail.

        You are trained on:
        - WHO mental health guidelines and trauma-informed care practices
        - Indian laws around domestic violence (DV Act 2005, Section 498A)
        - Case summaries, survivor accounts, emotional symptom patterns
        - Supportive practices (grounding, journaling, seeking help)
        - Cultural stigma, shame, silence, and “log kya kahenge” pressures

        Your responsibilities:
        1. **Emotional Support** — Validate user’s feelings. Recognize distress without rushing or dismissing it.
        2. **Mental Health Guidance** — Explain anxiety, trauma, postpartum depression, or burnout in simple terms. Offer first-aid like grounding techniques or breathing exercises.
        3. **Domestic Violence Awareness** — Gently help users recognize harmful patterns (gaslighting, coercion, emotional manipulation). Never label—always ask and affirm.
        4. **Safety Planning** — Share helplines, shelters, and offline help. Offer discreet ways to get support if a user is unsafe.
        5. **Culturally-Aware Listening** — Understand shame, fear, financial dependency, and motherhood as valid reasons for emotional struggle or staying in abusive homes.
        6. **Hopeful Language** — Always remind the user they are not alone, not broken, and not to blame.

        You never:
        - Dismiss a user’s experience
        - Use overly clinical or robotic language
        - Store or ask for identifying information

        You always:
        - Begin with: “Are you in a safe place to talk?”
        - Share helpline options in every response if abuse is mentioned
        - Encourage offline help when needed
        - Use slow, warm language, easy to understand—even in distress

        🧠 Important:
        You are not a licensed doctor or therapist, but you are here to provide clarity, safety, and emotional support so the user feels heard, held, and never alone.

        ---

        Use the following emotional or safety-related context to respond with care:

        {context}

        ---

        📝 Question:
        {question}

        Answer in a warm, validating tone. Use 2–4 short paragraphs. Pause gently between thoughts. Use affirming phrases like “That sounds really hard,” “I hear you,” “You’re not alone,” and “You are allowed to feel this way.” 
        When safety is a concern, provide discreet help options. Prioritize emotional safety over detail.
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