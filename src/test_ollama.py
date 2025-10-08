import requests
import json
from typing import Dict, Any


def check_ollama_status() -> bool:
    """Ollama servisinin çalıştığını kontrol et"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama çalışıyor!")
            print(f"Yüklü modeller: {[m['name'] for m in models.get('models', [])]}")
            return True
        else:
            print("❌ Ollama bağlantı sorunu")
            return False
    except Exception as e:
        print(f"❌ Ollama çalışmıyor: {e}")
        print("Terminal'de 'ollama serve' komutunu çalıştırın")
        return False


def ask_ollama(prompt: str, model: str = "llama3.1:8b-instruct-q4_K_M") -> Dict[str, Any]:
    """Ollama'ya soru sor"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response": result['response'],
                "model": model
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Test scriptini çalıştır"""
    print("="*60)
    print("OLLAMA TEST SCRIPTI")
    print("="*60 + "\n")
    
    # 1. Servis kontrolü
    if not check_ollama_status():
        return
    
    print("\n" + "="*60)
    print("TEST 1: Basit Soru")
    print("="*60)
    
    # 2. Basit test
    result = ask_ollama("What is Dungeons & Dragons? Answer in 2 sentences.")
    
    if result["success"]:
        print(f"✅ Cevap alındı!")
        print(f"\nSoru: What is Dungeons & Dragons?")
        print(f"Cevap: {result['response']}\n")
    else:
        print(f"❌ Hata: {result['error']}")
        return
    
    print("="*60)
    print("TEST 2: DnD Spesifik Soru")
    print("="*60)
    
    # 3. DnD spesifik test
    dnd_question = "What are the core ability scores in D&D 5e?"
    result = ask_ollama(dnd_question)
    
    if result["success"]:
        print(f"Soru: {dnd_question}")
        print(f"Cevap: {result['response']}\n")
    
    print("="*60)
    print("✅ Tüm testler tamamlandı!")
    print("="*60)


if __name__ == "__main__":
    main()