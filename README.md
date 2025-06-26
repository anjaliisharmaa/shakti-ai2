# SHAKTI-AI: Reproductive & Legal Health AI Agents

SHAKTI-AI is a suite of AI-powered agents designed to support Indian women and families with reproductive health, legal rights, mental wellness, and more. Each agent is tailored for a specific domain, providing empathetic, evidence-based, and culturally relevant guidance.

## Features
- **Maaya**: Maternal health nurse for pregnancy, childbirth, and baby care.
- **Gynika**: Reproductive health advisor for periods, puberty, and contraception.
- **Meher**: Emotional support counselor for trauma, anxiety, and abuse.
- **Nyaya**: Legal-ethical guide for Indian laws on consent, abortion, and family rights.
- **Vaanya**: Feminist educator for menopause and hormonal health.

## Tech Stack
- Python, Streamlit
- CrewAI, LangChain, LiteLLM, Google Gemini (Generative AI)
- FAISS for vector search
- PDF parsing and OCR for ingesting medical/legal documents

## Setup Instructions
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd shakti-ai
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Create a `.env` file in the root directory:
     ```env
     GOOGLE_API_KEY=your_google_gemini_api_key
     ```
4. **Run the app**
   ```sh
   streamlit run app.py
   ```
5. **Access the app**
   - Open your browser at [http://localhost:8501](http://localhost:8501)

## Project Structure
```
├── app.py                # Streamlit app entry point
├── agents.py             # CrewAI agent definitions
├── tools/                # LLM and agent tool wrappers
├── agentss/              # Core logic for each agent
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
└── README.md             # Project documentation
```

## Notes
- All LLM calls use Google Gemini via LiteLLM for privacy and cost control.
- No OpenAI API key is required or used.
- For best results, use a valid Google Gemini API key.

## License
MIT License

---
SHAKTI-AI: Empowering women with knowledge, empathy, and legal clarity.
