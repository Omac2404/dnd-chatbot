"""
Tüm sistemleri test eden entegrasyon scripti
"""

import requests
from anthropic import Anthropic
from config import config
from pdf_processor import list_pdfs


def test_all_systems():
    """Tüm sistemleri test et"""
    print("="*60)
    print("SİSTEM ENTEGRASYON TESTİ")
    print("="*60 + "\n")
    
    results = {}
    
    # 1. Python & Config
    print("1. Python & Config test ediliyor...")
    try:
        config.validate()
        results["config"] = True
        print("   ✅ Config hazır")
    except Exception as e:
        results["config"] = False
        print(f"   ❌ Config hatası: {e}")
    
    # 2. Ollama
    print("\n2. Ollama test ediliyor...")
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        results["ollama"] = response.status_code == 200
        if results["ollama"]:
            print("   ✅ Ollama çalışıyor")
        else:
            print("   ❌ Ollama bağlantı sorunu")
    except:
        results["ollama"] = False
        print("   ❌ Ollama çalışmıyor")
    
    # 3. Claude API
    print("\n3. Claude API test ediliyor...")
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}]
        )
        results["claude"] = True
        print("   ✅ Claude API çalışıyor")
    except Exception as e:
        results["claude"] = False
        print(f"   ❌ Claude API hatası: {e}")
    
    # 4. PDF'ler
    print("\n4. PDF'ler kontrol ediliyor...")
    pdfs = list_pdfs()
    results["pdf_count"] = len(pdfs)
    results["pdf"] = len(pdfs) > 0
    if results["pdf"]:
        print(f"   ✅ {len(pdfs)} PDF bulundu")
    else:
        print("   ⚠️ PDF bulunamadı")
    
    # 5. Gerekli kütüphaneler
    print("\n5. Kütüphaneler kontrol ediliyor...")
    try:
        import fitz  # type: ignore  # PyMuPDF
        import anthropic
        import requests
        results["libraries"] = True
        print("   ✅ Tüm kütüphaneler kurulu")
    except ImportError as e:
        results["libraries"] = False
        print(f"   ❌ Eksik kütüphane: {e}")
    
    # Özet
    print("\n" + "="*60)
    print("TEST SONUÇLARI")
    print("="*60)
    
    all_ok = all([
        results.get("config"),
        results.get("ollama"),
        results.get("claude"),
        results.get("pdf"),
        results.get("libraries")
    ])
    
    if all_ok:
        print("\n🎉 TÜM SİSTEMLER HAZIR!")
        print("✅ Hafta 2'ye geçebilirsiniz!")
    else:
        print("\n⚠️ Bazı sistemlerde sorun var:")
        for key, value in results.items():
            if not value:
                print(f"   ❌ {key}")
    
    return results


if __name__ == "__main__":
    test_all_systems()