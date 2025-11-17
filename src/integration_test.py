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
        import requests as req
        response = req.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        results["ollama"] = response.status_code == 200
        if results["ollama"]:
            print("   ✅ Ollama çalışıyor")
        else:
            print("   ❌ Ollama bağlantı sorunu")
    except Exception as e:
        results["ollama"] = False
        print(f"   ❌ Ollama çalışmıyor: {e}")
    
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
    try:
        pdfs = list_pdfs()
        results["pdf_count"] = len(pdfs)
        results["pdf"] = len(pdfs) > 0
        if results["pdf"]:
            print(f"   ✅ {len(pdfs)} PDF bulundu")
        else:
            print("   ⚠️ PDF bulunamadı")
    except Exception as e:
        results["pdf"] = False
        print(f"   ❌ PDF hatası: {e}")
    
    # 5. Gerekli kütüphaneler
    print("\n5. Kütüphaneler kontrol ediliyor...")
    try:
        import pymupdf as fitz  # type: ignore
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
        print("\n📊 Özet:")
        print(f"   • Config: ✅")
        print(f"   • Ollama: ✅")
        print(f"   • Claude API: ✅")
        print(f"   • PDF'ler: ✅ ({results.get('pdf_count', 0)} adet)")
        print(f"   • Kütüphaneler: ✅")
    else:
        print("\n⚠️ Bazı sistemlerde sorun var:")
        if not results.get("config"):
            print("   ❌ Config")
        if not results.get("ollama"):
            print("   ❌ Ollama")
        if not results.get("claude"):
            print("   ❌ Claude API")
        if not results.get("pdf"):
            print("   ❌ PDF'ler")
        if not results.get("libraries"):
            print("   ❌ Kütüphaneler")
    
    return results


if __name__ == "__main__":
    test_all_systems()