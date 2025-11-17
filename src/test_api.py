"""
API test scripti
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Health check"""
    r = requests.get(f"{BASE_URL}/health")
    print("✅ Health:", r.json())

def test_query():
    """Soru sor"""
    data = {
        "question": "What are ability scores?",
        "top_k": 5,
        "use_web": True
    }
    
    r = requests.post(f"{BASE_URL}/query", json=data)
    response = r.json()
    
    print("\n📝 Soru:", data['question'])
    print("💬 Cevap:", response['answer'][:200], "...")
    print("📊 Confidence:", response['confidence'])
    print("🔧 Method:", response['method_used'])
    print(f"⏱️ Response Time: {response['response_time']:.2f}s")

if __name__ == "__main__":
    print("="*60)
    print("API TEST")
    print("="*60)
    
    test_health()
    test_query()
    
    print("\n✅ Testler tamamlandı!")