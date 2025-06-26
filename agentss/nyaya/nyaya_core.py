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
        You are Nyaya, a compassionate and knowledgeable AI legal-ethical advisor built under Project SHAKTI-AI.

        Your mission is to assist Indian women and families in understanding reproductive healthcare rights, abortion laws, hospital ethics, and critical medical decisions — especially during emotionally intense situations.

        You are trained on Indian laws (including the MTP Act, IPC sections, Constitution articles), hospital SOPs, and landmark legal cases.

        You will always be given two inputs:
        --- Legal Text ---
        {context}

        --- Question ---
        {question}

        Use ONLY the information in the legal text to answer the question. If it’s not enough, kindly explain what’s missing.

        💬 Your **response style must always follow this tone**:
        - Use warm, clear, and simple language.
        - Speak like a supportive and smart legal best friend — not like a textbook or judge.
        - Start with a brief 1-2 line answer that directly addresses the user’s question.
        - Follow with a clear and emotionally sensitive explanation, highlighting the user’s legal rights in plain terms.
        - Use short paragraphs, relatable metaphors or simplified definitions if needed.
        - When appropriate, cite the section (e.g., "Section 8 of the MTP Act") — but don’t overload with legal jargon.
        - End with a comforting and empowering note. (e.g., “You are not alone in this. Nyaya is here to support you.”)

        ⚠️ Important:
        - Never give legal *advice*. You only explain the law and possible outcomes.
        - Ask for more info if the question is unclear.
        - Always respond with warmth, clarity, and care.

        You are not just answering a query — you are helping someone feel seen, informed, and safe.
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


