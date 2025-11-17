from typing import List, Dict
from vector_db import VectorDB
from embedder import Embedder
import requests
from anthropic import Anthropic
from config import config


class RAGPipeline:
    """RAG pipeline sınıfı"""
    
    def __init__(self, use_local_llm: bool = True):
        """
        Args:
            use_local_llm: True -> Llama (local), False -> Claude API
        """
        self.use_local_llm = use_local_llm
        
        # Vector DB
        print("📚 Vector database yükleniyor...")
        self.vector_db = VectorDB(collection_name="dnd_knowledge")
        
        # ÖNEMLİ: Database ile aynı model kullan (768-dim)
        print("🔤 Embedding modeli yükleniyor...")
        self.embedder = Embedder(model_name="all-mpnet-base-v2")
        
        # Claude client (fallback için)
        if not use_local_llm:
            self.claude_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        print(f"✅ RAG Pipeline hazır! LLM: {'Llama (Local)' if use_local_llm else 'Claude API'}")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query'ye alakalı context'leri vector DB'den al
        
        Args:
            query: Kullanıcı sorusu
            top_k: Kaç chunk döndürülsün
            
        Returns:
            En alakalı chunk'lar
        """
        print(f"🔍 Retrieval: '{query}' için {top_k} chunk aranıyor...")
        
        # ÖNEMLİ: Query'yi embedding'e çevir (aynı model ile!)
        query_embedding = self.embedder.embed_text(query)
        
        # Embedding ile ara
        results = self.vector_db.search(query, n_results=top_k, query_embedding=query_embedding)
        
        print(f"✅ {len(results)} alakalı chunk bulundu")
        return results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Retrieved document'ları prompt için formatlı string'e çevir"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            chunk_id = doc['metadata'].get('chunk_id', 'N/A')
            text = doc['text']
            
            # ✅ YENİ: Similarity score ekle (LLM'e hangi source'un daha önemli olduğunu gösterir)
            similarity = doc.get('similarity', 0.0)
            
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_id}, Relevance: {similarity:.2f}]\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_with_llama(self, query: str, context: str) -> str:
        """
        Llama (local) ile cevap üret
        
        Args:
            query: Kullanıcı sorusu
            context: Retrieved context
            
        Returns:
            Llama'nın cevabı
        """
        # ✅ İYİLEŞTİRİLMİŞ PROMPT
        prompt = f"""You are a D&D 5th Edition expert assistant. Your job is to answer questions using ONLY the provided context from the Player's Handbook.

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the context doesn't contain enough information, explicitly say: "I don't have enough information in the provided text to answer this question."
3. ALWAYS cite your sources using the format: (Source X: Chunk Y)
4. Be specific and detailed in your answer
5. If multiple sources provide relevant information, combine them coherently

Context from D&D Player's Handbook:
{context}

User Question: {query}

Answer:"""
        
        # Ollama API'ye istek
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        data = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512  # max_tokens yerine
            }
        }
        
        print("🤖 Llama ile cevap üretiliyor...")
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def generate_with_claude(self, query: str, context: str) -> str:
        """
        Claude API ile cevap üret
        
        Args:
            query: Kullanıcı sorusu
            context: Retrieved context
            
        Returns:
            Claude'un cevabı
        """
        prompt = f"""You are a D&D 5th Edition expert assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Provide a clear, accurate answer with source citations in the format (Source X: Chunk Y)."""
        
        print("🤖 Claude API ile cevap üretiliyor...")
        message = self.claude_client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def query(self, user_question: str, top_k: int = 5) -> Dict:
        """
        RAG pipeline'ın ana fonksiyonu
        
        Args:
            user_question: Kullanıcı sorusu
            top_k: Kaç context chunk kullanılsın (✅ Default 5'e çıkarıldı)
            
        Returns:
            Dict with 'answer', 'sources', 'context'
        """
        print("\n" + "="*60)
        print(f"📝 Soru: {user_question}")
        print("="*60)
        
        # 1. RETRIEVAL
        retrieved_docs = self.retrieve_context(user_question, top_k=top_k)
        
        # 2. FORMAT CONTEXT
        context = self.format_context(retrieved_docs)
        
        # 3. GENERATION
        if self.use_local_llm:
            answer = self.generate_with_llama(user_question, context)
        else:
            answer = self.generate_with_claude(user_question, context)
        
        print("✅ Cevap üretildi!")
        
        # Sonuç
        result = {
            "question": user_question,
            "answer": answer,
            "sources": [
                {
                    "source": doc['metadata']['source'],
                    "chunk_id": doc['metadata']['chunk_id'],
                    "text_preview": doc['text'][:200],
                    "similarity": doc.get('similarity', 0.0)
                }
                for doc in retrieved_docs
            ],
            "context_used": context
        }
        
        return result
    
    def calculate_confidence(self, answer: str, sources: List[Dict] = None) -> float:
        """
        ✅ İYİLEŞTİRİLMİŞ: Cevabın confidence skorunu hesapla
        
        Args:
            answer: LLM'in cevabı
            sources: Kullanılan kaynaklar (similarity skorları için)
        
        Returns:
            0.0 - 1.0 arası confidence score
        """
        confidence = 1.0  # Başlangıç: yüksek confidence
        
        # 1. Düşük confidence ifadeleri (ağırlık: -0.6)
        low_confidence_phrases = [
            "i don't know",
            "i don't have",
            "not sure",
            "unclear",
            "cannot find",
            "not enough information"
        ]
        
        answer_lower = answer.lower()
        
        for phrase in low_confidence_phrases:
            if phrase in answer_lower:
                confidence -= 0.6
                break
        
        # 2. Uzunluk kontrolü (çok kısa = şüpheli)
        if len(answer) < 50:
            confidence -= 0.3
        
        # 3. Source citation var mı? (önemli!)
        if "source" in answer_lower or "chunk" in answer_lower:
            confidence += 0.1  # Kaynak gösterdiyse +bonus
        else:
            confidence -= 0.2  # Kaynak göstermediyse -ceza
        
        # 4. ✅ YENİ: Retrieved sources'ların avg similarity
        if sources:
            avg_similarity = sum(s.get('similarity', 0.0) for s in sources) / len(sources)
            
            # Yüksek similarity = yüksek confidence
            if avg_similarity > 0.7:
                confidence += 0.1
            elif avg_similarity < 0.4:
                confidence -= 0.2
        
        # 5. Sınırla 0-1 arası
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence


def main():
    """RAG Pipeline test scripti"""
    
    print("="*60)
    print("RAG PIPELINE TEST")
    print("="*60 + "\n")
    
    # RAG pipeline oluştur
    rag = RAGPipeline(use_local_llm=True)  # Llama kullan
    
    # Test soruları
    test_questions = [
        "What are the six ability scores in D&D 5e?",
        "How do I calculate my armor class?",
        "What is a saving throw and when do I make one?",
        "Explain the difference between a skill check and an ability check.",
    ]
    
    print("\n" + "="*60)
    print("TEST SORULARI")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'#'*60}")
        
        # ✅ TOP_K = 5 (3 yerine)
        result = rag.query(question, top_k=5)
        
        # Sonuçları göster
        print("\n" + "-"*60)
        print("CEVAP:")
        print("-"*60)
        print(result['answer'])
        
        print("\n" + "-"*60)
        print("KAYNAKLAR:")
        print("-"*60)
        for j, source in enumerate(result['sources'], 1):
            print(f"\n{j}. {source['source']} (Chunk {source['chunk_id']})")
            print(f"   Similarity: {source['similarity']:.4f}")
            print(f"   Önizleme: {source['text_preview']}...")
        
        # ✅ İyileştirilmiş confidence calculation
        confidence = rag.calculate_confidence(result['answer'], result['sources'])
        print(f"\n📊 Confidence Score: {confidence:.2f}")
        
        if confidence < config.CONFIDENCE_THRESHOLD:
            print("⚠️ Düşük confidence - Web araması önerilir")
    
    print("\n\n" + "="*60)
    print("✅ RAG PIPELINE TESTİ TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    main()

