"""
Cache'li RAG Pipeline
"""
from rag_pipeline_hybrid import HybridRAGPipeline
from cache_manager import CacheManager
from typing import Dict

class CachedRAGPipeline(HybridRAGPipeline):
    """Cache mekanizmalı RAG pipeline"""
    
    def __init__(self):
        super().__init__()
        self.cache = CacheManager()
        print("💾 Cache yöneticisi hazır")
    
    def query(self, user_question: str, top_k: int = 5) -> Dict:
        """Cache kontrolü + RAG query"""
        
        # Cache kontrolü
        cached_result = self.cache.get(user_question)
        
        if cached_result:
            print("📦 Cache'den alındı!")
            return cached_result['result']
        
        # Cache'de yok, normal RAG
        result = super().query(user_question, top_k)
        
        # Cache'e kaydet
        self.cache.set(user_question, result)
        
        return result