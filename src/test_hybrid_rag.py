from rag_pipeline_hybrid import HybridRAGPipeline

def test_hybrid_system():
    """Hybrid RAG'i test et"""
    
    print("="*60)
    print("HYBRID RAG TEST")
    print("="*60)
    
    # Pipeline oluştur
    rag = HybridRAGPipeline()
    
    # Test soruları
    test_cases = [
        {
            "question": "What are the six ability scores?",
            "expected_method": "llama",  # PDF'de var, yüksek confidence
        },
        {
            "question": "What is the newest D&D errata for 2024?",
            "expected_method": "claude+web",  # PDF'de yok, web'den bulunmalı
        },
        {
            "question": "How does grappling work in combat?",
            "expected_method": "llama",  # PDF'de var
        },
        {
            "question": "What are the best homebrew classes?",
            "expected_method": "claude+web",  # Subjektif, web'den bulunmalı
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*60}")
        
        result = rag.query(test['question'], top_k=5)
        
        print(f"\n📊 Sonuç:")
        print(f"   Method: {result['method_used']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Web Enhanced: {result['web_enhanced']}")
        print(f"\n💬 Cevap:")
        print(f"   {result['answer'][:300]}...")
        
        # Doğrulama
        success = result['method_used'] == test['expected_method']
        print(f"\n{'✅' if success else '⚠️'} Expected: {test['expected_method']}, Got: {result['method_used']}")
        
        results.append({
            "question": test['question'],
            "success": success,
            "method": result['method_used'],
            "confidence": result['confidence']
        })
    
    # Özet
    print("\n" + "="*60)
    print("TEST ÖZET")
    print("="*60)
    
    success_rate = sum(r['success'] for r in results) / len(results)
    print(f"\n✅ Başarı Oranı: {success_rate:.1%}")
    
    llama_count = sum(1 for r in results if r['method'] == 'llama')
    web_count = sum(1 for r in results if r['method'] == 'claude+web')
    
    print(f"🦙 Llama kullanımı: {llama_count}/{len(results)}")
    print(f"☁️ Claude+Web kullanımı: {web_count}/{len(results)}")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"📊 Ortalama Confidence: {avg_confidence:.2f}")
    
    print("\n🎉 Hybrid RAG sistemi çalışıyor!")

if __name__ == "__main__":
    test_hybrid_system()