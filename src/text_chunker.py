"""
Text chunking modülü
PDF text'ini parçalara böler ve metadata ekler
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from typing import List, Dict
import re


class TextChunker:
    """Text'i anlamlı parçalara bölen sınıf"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Her chunk'ın maksimum karakter sayısı
            chunk_overlap: Chunk'lar arası örtüşme (context sürekliliği için)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter: Paragraf > Cümle > Kelime sırasında böler
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Öncelik sırası
        )
    
    def clean_text(self, text: str) -> str:
        """Text'i temizle (geliştirilmiş)"""
        # Harf arası fazla boşlukları temizle (OCR hatası)
        text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text, flags=re.IGNORECASE)
        
        # Kelime arası çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Fazla newline'ları temizle
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Garip karakterleri temizle
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\'\"]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, source_name: str = "unknown") -> List[Dict]:
        """
        Text'i chunk'lara böl ve metadata ekle
        
        Args:
            text: Bölünecek text
            source_name: PDF dosya adı (metadata için)
            
        Returns:
            List of dicts with 'text' and 'metadata'
        """
        # Text'i temizle
        cleaned_text = self.clean_text(text)
        
        # Chunk'lara böl
        chunks = self.splitter.split_text(cleaned_text)
        
        # Her chunk'a metadata ekle
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "metadata": {
                    "source": source_name,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk)
                }
            }
            chunked_documents.append(doc)
        
        return chunked_documents
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Text'ten basit keyword extraction (gelişmiş versiyonlar için)"""
        # Basit versiyon: En uzun kelimeler
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # En sık geçen kelimeleri al
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:max_keywords]]
        
        return keywords


def main():
    """Test scripti"""
    from pdf_processor import extract_text_from_pdf, list_pdfs
    
    print("="*60)
    print("TEXT CHUNKING TEST")
    print("="*60 + "\n")
    
    # PDF'leri listele
    pdfs = list_pdfs()
    
    if not pdfs:
        print("❌ PDF bulunamadı!")
        return
    
    # İlk PDF'i kullan
    pdf_path = pdfs[0]
    print(f"📄 Test PDF: {pdf_path.name}\n")
    
    # Text çıkar
    print("📖 Text çıkarılıyor...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✅ Toplam {len(text):,} karakter\n")
    
    # Chunker oluştur
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    
    # Chunk'lara böl
    print("✂️ Text chunk'lara bölünüyor...")
    chunks = chunker.chunk_text(text, source_name=pdf_path.name)
    print(f"✅ {len(chunks)} chunk oluşturuldu\n")
    
    # İlk 3 chunk'ı göster
    print("="*60)
    print("İLK 3 CHUNK ÖRNEĞİ")
    print("="*60)
    
    for i in range(min(3, len(chunks))):
        chunk = chunks[i]
        print(f"\n--- Chunk {i+1} ---")
        print(f"Kaynak: {chunk['metadata']['source']}")
        print(f"Chunk ID: {chunk['metadata']['chunk_id']}/{chunk['metadata']['total_chunks']}")
        print(f"Karakter: {chunk['metadata']['char_count']}")
        print(f"İçerik: {chunk['text'][:200]}...")
    
    # İstatistikler
    print("\n" + "="*60)
    print("CHUNK İSTATİSTİKLERİ")
    print("="*60)
    
    chunk_sizes = [chunk['metadata']['char_count'] for chunk in chunks]
    print(f"Toplam chunk: {len(chunks)}")
    print(f"Ortalama boyut: {sum(chunk_sizes) / len(chunk_sizes):.0f} karakter")
    print(f"Min boyut: {min(chunk_sizes)} karakter")
    print(f"Max boyut: {max(chunk_sizes)} karakter")
    
    print("\n✅ Text chunking başarılı!")


if __name__ == "__main__":
    main()