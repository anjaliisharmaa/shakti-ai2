from nyaya_core import get_all_pdf_text, get_text_chunks
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("🔍 Reading PDFs...")
raw_text = get_all_pdf_text("nyaya_raw_data")

print("✂️ Splitting text into chunks...")
chunks = get_text_chunks(raw_text)
print(f"✅ Total Chunks: {len(chunks)}")

print("🧠 Generating embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

# Embed chunks in smaller batches to show progress
batched_chunks = [chunks[i:i+10] for i in range(0, len(chunks), 10)]
all_chunks = []
for i, batch in enumerate(batched_chunks):
    print(f"💡 Embedding batch {i+1}/{len(batched_chunks)}")
    all_chunks.extend(batch)  # collecting for final FAISS build

# Only now: build full vector store
vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)

print("💾 Saving vector store...")
vector_store.save_local("faiss_index")

print("✅ Vector store created and saved successfully.")
