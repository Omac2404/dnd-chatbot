"""
Proje konfigürasyon ayarları ollama + antropic
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class Config:
    """Proje konfigürasyon sınıfı"""
    
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
    
    # RAG Settings
    CHUNK_SIZE = 512  
    CHUNK_OVERLAP = 50  
    TOP_K = 7  
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
    VECTOR_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
    
    # Claude Settings
    CLAUDE_MODEL = "claude-3-5-haiku-20241022"
    CONFIDENCE_THRESHOLD = 0.8
    
    @classmethod
    def validate(cls):
        """Konfigürasyonu doğrula"""
        errors = []
        
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY bulunamadı (.env dosyasını kontrol edin)")
        
        if not cls.PDF_DIR.exists():
            cls.PDF_DIR.mkdir(parents=True, exist_ok=True)
            print(f"📁 {cls.PDF_DIR} klasörü oluşturuldu")
        
        if not cls.VECTOR_DB_DIR.exists():
            cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            print(f"📁 {cls.VECTOR_DB_DIR} klasörü oluşturuldu")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return True


# Config'i validate et
config = Config()

if __name__ == "__main__":
    print("Config Ayarları:")
    print(f"Claude Model: {config.CLAUDE_MODEL}")
    print(f"Ollama Model: {config.OLLAMA_MODEL}")
    print(f"PDF Directory: {config.PDF_DIR}")
    print(f"API Key var mı: {'✅' if config.ANTHROPIC_API_KEY else '❌'}")
    
    try:
        config.validate()
        print("\n✅ Tüm konfigürasyon ayarları doğru!")
    except ValueError as e:
        print(f"\n❌ Konfigürasyon hatası:\n{e}")