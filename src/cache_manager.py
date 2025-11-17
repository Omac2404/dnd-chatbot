"""
Cache yönetimi - Sık sorulan sorular için kontrol edilmesi lazım!!
"""
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd # type: ignore
import hashlib

class CacheManager:
    """Soru-cevap cache yönetimi"""
    
    def __init__(self, cache_file: str = "data/query_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Cache dosyasını yükle"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Cache'i diske kaydet"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)
    
    def _hash_query(self, query: str) -> str:
        """Query'den hash oluştur"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Cache'den getir"""
        query_hash = self._hash_query(query)
        return self.cache.get(query_hash)
    
    def set(self, query: str, result: Dict):
        """Cache'e kaydet"""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {
            'query': query,
            'result': result,
            'timestamp': str(pd.Timestamp.now())
        }
        self._save_cache()
    
    def clear(self):
        """Cache'i temizle"""
        self.cache = {}
        self._save_cache()