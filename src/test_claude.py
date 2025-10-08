"""
Claude API test scripti
"""

from anthropic import Anthropic
from config import config


def test_claude_connection():
    """Claude API bağlantısını test et"""
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hi, respond with just 'Hello!'"}
            ]
        )
        
        print("✅ Claude API çalışıyor!")
        return True
    except Exception as e:
        print(f"❌ Claude API hatası: {e}")
        return False


def ask_claude(prompt: str):
    """Claude'a soru sor"""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "answer": message.content[0].text,
        "tokens": message.usage.input_tokens + message.usage.output_tokens
    }


def main():
    """Test scriptini çalıştır"""
    print("="*60)
    print("CLAUDE API TEST SCRIPTI")
    print("="*60 + "\n")
    
    # 1. Bağlantı testi
    if not test_claude_connection():
        return
    
    print("\n" + "="*60)
    print("DnD Sorusu Testi")
    print("="*60)
    
    # 2. DnD sorusu
    question = "What is a D&D saving throw? Answer briefly."
    result = ask_claude(question)
    
    print(f"Soru: {question}")
    print(f"\nCevap: {result['answer']}")
    print(f"\nKullanılan token: {result['tokens']}")
    print(f"Tahmini maliyet: ${result['tokens'] * 0.80 / 1_000_000:.6f}")
    
    print("\n" + "="*60)
    print("✅ Test tamamlandı!")
    print("="*60)


if __name__ == "__main__":
    main()