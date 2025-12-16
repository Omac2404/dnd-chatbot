from typing import List, Dict
from vector_db import VectorDB
from embedder import Embedder
from web_scraper import WebScraper
import requests
from anthropic import Anthropic
from config import config

class HybridRAGPipeline:
    """Geliştirilmiş RAG pipeline - fallback mekanizmalı"""
    
    def __init__(self):
        """Pipeline'ı initialize et"""
        print("🚀 Hybrid RAG Pipeline başlatılıyor...")
        
        # Vector DB ve Embedder
        self.vector_db = VectorDB(collection_name="dnd_knowledge")
        self.embedder = Embedder(model_name="all-mpnet-base-v2")
        
        # Web Scraper
        self.web_scraper = WebScraper()
        
        # Claude client
        self.claude_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        print("✅ Hybrid RAG hazır!")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector DB'den context al"""
        print(f"🔍 Retrieval: '{query}'")
        
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_db.search(
            query, 
            n_results=top_k, 
            query_embedding=query_embedding
        )
        
        print(f"✅ {len(results)} chunk bulundu")
        return results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Context'i formatlı string'e çevir"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            chunk_id = doc['metadata'].get('chunk_id', 'N/A')
            similarity = doc.get('similarity', 0.0)
            
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_id}, Similarity: {similarity:.2f}]\n{doc['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_with_llama(self, query: str, context: str) -> str:
        """Llama ile cevap üret"""
        prompt = f"""You are a D&D 5th Edition expert. Answer ONLY based on the provided context.

Context:
{context}

Question: {query}

Answer with source citations (Source X):"""
        
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        data = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        print("🦙 Llama ile cevap üretiliyor...")
        response = requests.post(url, json=data, timeout=300)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama hatası: {response.status_code}")
    
    def generate_with_claude(self, query: str, context: str, web_context: str = None) -> str:
        """Claude ile cevap üret (fallback)"""
        
        if web_context:
            prompt = f"""You are a D&D expert. Answer using BOTH the PDF context and web search results.

PDF Context:
{context}

Web Search Results:
{web_context}

Question: {query}

Provide a comprehensive answer citing sources."""
        else:
            prompt = f"""You are a D&D expert. Answer based on the context.

Context:
{context}

Question: {query}

Answer with citations."""
        
        print("☁️ Claude ile cevap üretiliyor...")
        message = self.claude_client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def calculate_confidence(self, answer: str, sources: List[Dict]) -> float:

        confidence = 0.5  # 1.0 yerine 0.5'ten başla (daha temkinli)
        
        # 1. Düşük confidence ifadeleri (ağırlık: -0.6)
        low_confidence_phrases = [
            "i don't know",
            "i don't have",
            "i'm not sure",
            "not sure",
            "unclear",
            "cannot find",
            "not enough information",
            "no information",
            "i cannot answer"
        ]
        
        answer_lower = answer.lower()
        
        for phrase in low_confidence_phrases:
            if phrase in answer_lower:
                confidence -= 0.6
                break
        
        # 2. Uzunluk kontrolü (çok kısa = şüpheli)
        if len(answer) < 50:
            confidence -= 0.3
        elif len(answer) > 100:
            confidence += 0.1  # Detaylı cevap +bonus
        
        # 3. Source citation var mı? (önemli!)
        if "source" in answer_lower or "chunk" in answer_lower:
            confidence += 0.15  # Kaynak gösterdiyse +bonus
        else:
            confidence -= 0.25  # Kaynak göstermediyse -ceza (artırıldı)
        
        # 4. Retrieved sources'ların avg similarity (ÇOK ÖNEMLİ!)
        if sources:
            avg_similarity = sum(s.get('similarity', 0.0) for s in sources) / len(sources)
            
            # Yüksek similarity = yüksek confidence
            if avg_similarity > 0.7:
                confidence += 0.2  # 0.1'den 0.2'ye çıkarıldı
            elif avg_similarity > 0.5:
                confidence += 0.1
            elif avg_similarity < 0.5:  # 0.4'ten 0.5'e yükseltildi
                confidence -= 0.4  # 0.2'den 0.4'e çıkarıldı (daha sıkı)
        else:
            confidence -= 0.5  # Kaynak yoksa büyük ceza
        
        # 5. Sınırla 0-1 arası
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def query(self, user_question: str, top_k: int = 5) -> Dict:
        """
        Hybrid RAG query - Llama -> Confidence Check -> Claude + Web Fallback
        
        Returns:
            Dict with answer, sources, confidence, method_used
        """
        print("\n" + "="*60)
        print(f"📝 Soru: {user_question}")
        print("="*60)
        
         # 0. Eğer Ollama devre dışı ise direkt Claude kullan
        if not config.USE_OLLAMA:
            # 1) PDF context al
            retrieved_docs = self.retrieve_context(user_question, top_k=top_k)
            context = self.format_context(retrieved_docs)

            # 2) Web araması
            web_results = self.web_scraper.search_dnd_content(user_question, max_results=3)
            web_context = self.web_scraper.format_web_results(web_results)

            # 3) Claude cevabı
            claude_answer = self.generate_with_claude(
                user_question,
                context,
                web_context=web_context,
            )

            confidence = self.calculate_confidence(claude_answer, retrieved_docs)
            confidence = min(confidence + 0.3, 1.0)  # Claude + web olduğu için biraz boost

            return {
                "question": user_question,
                "answer": claude_answer,
                "confidence": confidence,
                "sources": [
                    {
                        "source": doc["metadata"]["source"],
                        "chunk_id": doc["metadata"]["chunk_id"],
                        "text_preview": doc["text"][:200],
                        "similarity": doc.get("similarity", 0.0),
                    }
                    for doc in retrieved_docs
                ],
                "web_sources": [
                    {"url": r["url"], "preview": r["text"][:200]} for r in web_results
                ],
                "method_used": "claude+web",
                "web_enhanced": True,
            }

        
        # 1. RETRIEVAL
        retrieved_docs = self.retrieve_context(user_question, top_k=top_k)
        context = self.format_context(retrieved_docs)
        
        # 2. LLAMA GENERATION
        llama_answer = self.generate_with_llama(user_question, context)
        
        # 3. CONFIDENCE CHECK
        confidence = self.calculate_confidence(llama_answer, retrieved_docs)
        print(f"📊 Confidence: {confidence:.2f}")
        
        # 4. FALLBACK DECISION
        if confidence >= config.CONFIDENCE_THRESHOLD:
            # ✅ Yüksek confidence - Llama cevabını kullan
            print("✅ Yüksek confidence - Llama cevabı kullanılıyor")
            
            return {
                "question": user_question,
                "answer": llama_answer,
                "confidence": confidence,
                "sources": [
                    {
                        "source": doc['metadata']['source'],
                        "chunk_id": doc['metadata']['chunk_id'],
                        "text_preview": doc['text'][:200],
                        "similarity": doc.get('similarity', 0.0)
                    }
                    for doc in retrieved_docs
                ],
                "method_used": "llama",
                "web_enhanced": False
            }
        
        else:
            # ⚠️ Düşük confidence - Web + Claude fallback
            print("⚠️ Düşük confidence - Web araması + Claude fallback")
            
            # Web search
            web_results = self.web_scraper.search_dnd_content(user_question, max_results=3)
            web_context = self.web_scraper.format_web_results(web_results)
            
            # Claude with web context
            claude_answer = self.generate_with_claude(
                user_question, 
                context, 
                web_context
            )
            
            # Yeni confidence (genellikle daha yüksek)
            new_confidence = min(confidence + 0.3, 1.0)
            
            return {
                "question": user_question,
                "answer": claude_answer,
                "confidence": new_confidence,
                "sources": [
                    {
                        "source": doc['metadata']['source'],
                        "chunk_id": doc['metadata']['chunk_id'],
                        "text_preview": doc['text'][:200],
                        "similarity": doc.get('similarity', 0.0)
                    }
                    for doc in retrieved_docs
                ],
                "web_sources": [
                    {"url": r['url'], "preview": r['text'][:200]}
                    for r in web_results
                ],
                "method_used": "claude+web",
                "web_enhanced": True
            }