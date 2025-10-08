# DnD RAG Chatbot

D&D kurallarını öğrenen ve soruları cevaplayabilen AI chatbot.

## 🚀 Kurulum

1. Clone repo
2. Virtual environment: `python -m venv venv`
3. Aktive et: `venv\Scripts\activate` (Win) / `source venv/bin/activate` (Mac/Linux)
4. Kütüphaneleri kur: `pip install -r requirements.txt`
5. Ollama kur: https://ollama.ai
6. Llama indir: `ollama pull llama3.1:8b-instruct-q4_K_M`
7. `.env` dosyası oluştur, API key ekle

## 📁 Proje Yapısı
dnd-chatbot/
├── src/              # Kaynak kodlar
├── data/
│   ├── pdfs/         # DnD PDF'leri
│   └── chroma_db/    # Vector database
├── tests/
├── .env              # API keys
└── requirements.txt

## 🧪 Testler
```bash
python src/config.py
python src/test_ollama.py
python src/test_claude.py
python src/pdf_processor.py
python src/integration_test.py