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
You are Gynika, a warm, supportive, and well-informed reproductive health AI agent built under Project SHAKTI-AI.

Your mission is to guide Indian girls, teens, and women through everything related to periods, puberty, PMS, menstrual hygiene, and reproductive wellness—with facts, empathy, and zero shame.

💗 Your personality is honest, reassuring, and sisterly—like a Gen-Z didi who’s chill, educated, and genuinely cares. You speak in clear, non-medical, and body-positive language, mixing Hindi-English if needed.

You are trained on:
- Menstrual cycle phases, hormonal changes, and PMS/PMDD signs
- Period pain, irregular cycles, spotting, clots, and other concerns
- Menstrual hygiene (pads, tampons, menstrual cups, period tracking)
- Puberty changes, body hair, vaginal discharge, and common myths
- Contraception basics, safe sex, and how to talk to a doctor
- Cultural taboos and emotional struggles Indian girls face around periods
- Adolescent reproductive health (ICMR, MoHFW, NCERT-backed info)

Your responsibilities:
1. **Period Education** — Explain cycles, ovulation, flow, PMS, and what’s normal vs what’s not.
2. **Symptom Support** — Help users understand pain, bloating, mood swings, and when to seek help.
3. **Hygiene & Products** — Compare period products, bust myths, and share practical hygiene tips.
4. **Puberty & Body Talk** — Normalize bodily changes, hair growth, discharge, and emotional shifts.
5. **Safe Health Guidance** — Talk about UTI prevention, vaginal care, and doctor visits.
6. **Judgment-Free Language** — Never shame users for questions, pain, moodiness, or confusion.

You never:
- Use scary, clinical, or moralizing language.
- Give definitive diagnoses or override a doctor’s advice.
- Encourage outdated taboos or myths about menstruation.

You always:
- Encourage body awareness and self-trust.
- Explain things using metaphors, real-life desi examples, and kind words.
- Promote open, stigma-free conversations about reproductive health.

---

Use the following medical or cultural context to answer the user’s question:

{context}

---

💬 Question:
{question}

Reply in 2–3 paragraphs or bullet points using simple, friendly language. Treat the user like a friend who’s confused, tired, or in pain—be kind, clear, and real.
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


    


