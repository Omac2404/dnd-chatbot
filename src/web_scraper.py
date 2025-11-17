import requests
from bs4 import BeautifulSoup # type: ignore
from typing import List, Dict
import time
from config import config

class WebScraper:
    """Web scraping sınıfı"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
    
    def search_dnd_content(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        D&D içeriği için web araması yap
        
        Args:
            query: Arama sorgusu
            max_results: Maksimum sonuç sayısı
            
        Returns:
            Web scraping sonuçları
        """
        # Cache kontrolü
        if query in self.cache:
            print(f"📦 Cache'den alındı: {query}")
            return self.cache[query]
        
        print(f"🌐 Web araması: {query}")
        
        
        search_urls = [
            f"https://www.dndbeyond.com/search?q={query.replace(' ', '+')}",
            f"https://roll20.net/compendium/dnd5e/{query.replace(' ', '%20')}",
        ]
        
        results = []
        
        for url in search_urls[:max_results]:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Text extraction (site'a göre özelleştirin)
                    text = soup.get_text(separator=' ', strip=True)
                    
                    # İlk 1000 karakteri al
                    text = text[:1000]
                    
                    results.append({
                        'url': url,
                        'text': text,
                        'source': 'web'
                    })
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Web scraping hatası: {e}")
                continue
        
        # Cache'e kaydet
        self.cache[query] = results
        
        print(f"✅ {len(results)} web sonucu bulundu")
        return results
    
    def format_web_results(self, results: List[Dict]) -> str:
        """Web sonuçlarını formatlı string'e çevir"""
        formatted = []
        
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[Web Source {i}: {result['url']}]\n{result['text']}"
            )
        
        return "\n\n---\n\n".join(formatted)