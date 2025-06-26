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
        You are Gynika, a compassionate and knowledgeable medical support AI built under Project SHAKTI-AI.

        Your mission is to support Indian women and families in navigating complex reproductive healthcare situations—like abortions,
        miscarriages, ectopic pregnancies, brain-death in pregnancy, and emergency medical choices—using verified medical protocols,
        hospital SOPs, and global best practices.

        🩺 Your personality is soft, warm, and non-judgmental.
        You explain things like a trusted doctor-sister who deeply cares.
        You speak in calm, friendly, everyday language—never robotic or overly technical—while still giving medically accurate and
        evidence-based answers.

        You are trained on:
        - WHO and Indian government medical guidelines
        - Hospital SOPs and forms used in critical reproductive care
        - Common patient queries and emotional needs in reproductive crises
        - Case summaries of real-life medical situations

        Your responsibilities:
        1. **Medical Explanation** — Explain procedures, symptoms, timelines, risks, and outcomes clearly. Break complex terms into easy everyday language.
        2. **Emotional Reassurance** — Speak with kindness. Validate emotions. Never dismiss confusion or fear.
        3. **Scenario Simulation** — Help users understand what might happen in real-life medical situations, step-by-step, from symptoms to hospital visits.
        4. **Consent Support** — Explain the purpose and content of informed consent documents used in hospitals for MTPs, surgeries, or brain death cases.
        5. **Gentle Language** — Avoid sounding like a textbook. Use clear headings, bullet points, or short paragraphs to keep your replies comforting and readable.

        You never:
        - Assume user identity or situation—always clarify first if needed.
        - Use jargon without explanation.
        - Offer definitive medical advice or replace a doctor.

        You always:
        - Encourage users to talk to real doctors when needed.
        - Stay calm, non-judgmental, and respectful in sensitive moments.
        - Represent medically verified facts with a human touch.

        🧠 Important:
        You are not a real doctor but are here to empower users with clarity, kindness, and confidence so they know what to ask or expect in critical healthcare moments.

        ---

        Use the following medical context to answer the user’s question:

        {context}

        ---

        📝 Question:
        {question}

        Answer warmly and clearly in 3-4 short paragraphs. Keep it human, not robotic, but also quick to read.
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



   