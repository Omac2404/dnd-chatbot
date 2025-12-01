# src/chat_memory.py

from tinydb import TinyDB # type: ignore
from pathlib import Path

# Proje kökünü bul (dnd-chatbot)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "chat_history.json"

db = TinyDB(DB_PATH)
history_table = db.table("history")


def get_chat_history():
    """
    Tüm chat kayıtlarını sıralı şekilde döndürür.
    Her kayıt, streamlit_app_advanced.py'deki gibi bir Q&A dict'idir:
    {
      'question': ...,
      'answer': ...,
      'confidence': ...,
      'method': ...,
      'sources': ...,
      'web_enhanced': ...,
      'web_sources': ...,
      'response_time': ...
    }
    """
    return history_table.all()


def save_message(entry: dict):
    """
    Tek bir soru-cevap kaydını DB'ye ekler.
    """
    history_table.insert(entry)


def clear_history():
    """
    Tüm chat geçmişini siler.
    """
    history_table.truncate()
