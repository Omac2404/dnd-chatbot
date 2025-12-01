# DnD RAG Chatbot

An AI-powered assistant that learns Dungeons & Dragons (D&D 5e) rules from PDFs, answers rules questions using a RAG (Retrieval-Augmented Generation) pipeline, and falls back to web search + a cloud LLM when local context is not enough.

> **Status:** Early development – core environment, local LLM connectivity, Claude API connectivity, and PDF processing scripts are set up.

---

## ✨ Features (Planned)

- 📚 **Rule-aware DnD assistant**  
  Answers rules questions using official D&D PDFs (Player’s Handbook, DMG, etc.) via a vector-search-backed RAG pipeline.

- 🧠 **Hybrid RAG system**
  - Local Llama 3.1 8B (via Ollama) as the primary model
  - Sentence-transformer embeddings stored in ChromaDB / FAISS
  - Optional re-ranking for higher-quality retrieval

- 🌐 **Web-enhanced fallback**
  - If confidence in the RAG answer is low, the system:
    1. Searches the web and scrapes relevant pages  
    2. Calls Claude (Haiku 4) with both the original question and fetched content  
    3. Returns an answer clearly marked as “web-enhanced”

- 🧱 **Modular architecture**
  - Clean separation between config, LLM tests, PDF processing, and future RAG pipeline
  - Designed to grow into a full FastAPI backend + simple web UI (Streamlit/Gradio)

- 🧪 **Evaluation & safety focus (planned)**
  - Test question set (50–100 questions)
  - Accuracy, response time, hallucination rate, and user feedback tracking

A detailed 8–10 week roadmap for this project (from setup to deployment) is documented in the project planning document. :contentReference[oaicite:0]{index=0}

---

## 🏗 Architecture Overview

High-level flow:

1. **User question** (e.g., “How does a saving throw work in 5e?”)
2. **Query preprocessing**
   - Spell check
   - Named entity extraction (spells, classes, conditions)
   - Optional query expansion
3. **Vector similarity search**
   - ChromaDB / FAISS
   - Top-k (3–5) most relevant chunks from D&D PDFs
   - Metadata filtering by book / chapter / page
4. **(Optional) Re-ranking**
   - Cross-encoder re-ranks retrieved chunks for better relevance
5. **Local LLM answer (Llama 3.1 8B via Ollama)**
   - Prompt template combines context + user question
   - Confidence score computed from answer heuristics
6. **Fallback decision**
   - If `confidence >= threshold` → return answer + PDF sources
   - Else:
     - Run web search + scraping
     - Ask Claude (Haiku 4) with combined context
     - Return “web-enhanced” answer + sources

---

## 🧰 Tech Stack

**LLMs & AI**
- Local: Llama 3.1 8B (via Ollama)
- Cloud: Claude Haiku 4 (Anthropic API)
- Embeddings: `sentence-transformers`
- RAG Framework: LangChain / LlamaIndex (planned)

**Storage**
- Vector DB: ChromaDB / FAISS
- File data: D&D rule PDFs under `data/pdfs/`
- Vector index under `data/chroma_db/`

**Backend & Tools**
- Python 3.10+
- FastAPI (planned)
- Streamlit / Gradio for prototype UI (planned)

**Utilities**
- PDF processing: PyMuPDF (`pymupdf`), `pypdf2`, `pdfplumber`
- Web scraping: `requests`, `beautifulsoup4`, Selenium
- Environment & config: `python-dotenv`, `Config` class in `src/config.py`

The first week of work focuses on VS Code setup, environment creation, LLM/API tests, and PDF processing, as outlined in the detailed Week 1 plan. :contentReference[oaicite:1]{index=1}

---

## 📁 Project Structure

```txt
dnd-chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py          # Project configuration & paths
│   ├── test_ollama.py     # Local Llama / Ollama connectivity test
│   ├── test_claude.py     # Claude API connectivity test
│   ├── pdf_processor.py   # PDF listing, metadata & text extraction
│   └── integration_test.py# End-to-end environment sanity check
├── data/
│   ├── pdfs/              # Place DnD rule PDFs here
│   └── chroma_db/         # Vector database files (Chroma / FAISS)
├── tests/                 # (Reserved for future unit/integration tests)
├── .env                   # API keys & environment variables (not committed)
├── .gitignore
├── README.md
└── requirements.txt
