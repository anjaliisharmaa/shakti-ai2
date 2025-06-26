import streamlit as st
from crew_setup import ask_shakti_ai


# Set up page
st.set_page_config(page_title="SHAKTI-AI", layout="wide")


# Sidebar with agent bios
st.sidebar.title("🧬 About SHAKTI-AI")
st.sidebar.markdown("""
**SHAKTI-AI** is a multi-agent AI system built to support Indian women across health, legal, mental, and social issues.  
Meet your 5 AI sisters 👇
---
👶 **Maaya** – Maternal & Baby Care  
💗 **Gynika** – Periods, Fertility & Reproductive Health  
🧠 **Meher** – Mental Health & Abuse Support  
⚖️ **Nyaya** – Legal & Ethical Advisor  
📢 **Vaanya** – Menopause & Hormonal Balance  
---
They collaborate to give you complete, compassionate answers.
""")

# Main title
st.title("🌸 SHAKTI-AI — Ask Us Anything")
st.markdown("A safe space for questions about your body, mind, rights, and life. We're here. All five of us 💪")

# User input
user_question = st.text_input("💬 What's on your mind?", placeholder="e.g. I'm scared to talk about my pregnancy at home...")

if st.button("✨ Get Support"):
    if not user_question.strip():
        st.warning("Please enter your question first.")
    else:
        with st.spinner("Your SHAKTI sisters are talking..."):
            try:
                response = ask_shakti_ai(user_question)
                st.markdown("### 💡 SHAKTI-AI says:")
                st.markdown(response)
            except Exception as e:
                st.error(f"Oops! Something went wrong: {e}")
