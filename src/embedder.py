"""
Embedding modülü
Text'i vektörlere çevirir (semantic search için)
"""

from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore
from typing import List, Dict
from tqdm import tqdm # type: ignore


class Embedder:
    """Text embedding sınıfı"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Kullanılacak embedding modeli
            
        Popüler modeller:
        - all-MiniLM-L6-v2: Hızlı, 384-dim (ÖNERİLEN)
        - all-mpnet-base-v2: Daha iyi, 768-dim (yavaş ama daha başarılı)
        - paraphrase-MiniLM-L6-v2: Paraphrase detection için
        """
        print(f"📥 Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model hazır! Embedding boyutu: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Tek bir text'i embedding'e çevir"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Birden fazla text'i batch olarak embedding'e çevir
        
        Args:
            texts: Text listesi
            batch_size: Batch boyutu (GPU yoksa 32 yeterli)
            show_progress: Progress bar göster mi?
            
        Returns:
            (N, embedding_dim) shaped numpy array
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk document'larına embedding ekle
        
        Args:
            documents: text_chunker'dan gelen documents
            
        Returns:
            Her document'a 'embedding' field'ı eklenmiş liste
        """
        texts = [doc['text'] for doc in documents]
        
        print(f"🔄 {len(texts)} chunk embedding'e çevriliyor...")
        embeddings = self.embed_batch(texts, show_progress=True)
        
        # Embedding'leri document'lara ekle
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        print(f"✅ Embedding tamamlandı!")
        return documents
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """İki embedding arasında cosine similarity hesapla"""
        # Cosine similarity = dot product / (norm1 * norm2)
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)


def main():
    """Test scripti"""
    from text_chunker import TextChunker
    from pdf_processor import extract_text_from_pdf, list_pdfs
    
    print("="*60)
    print("EMBEDDING TEST")
    print("="*60 + "\n")
    
    # PDF'leri listele
    pdfs = list_pdfs()
    
    if not pdfs:
        print("❌ PDF bulunamadı!")
        return
    
    pdf_path = pdfs[0]
    print(f"📄 Test PDF: {pdf_path.name}\n")
    
    # Text çıkar ve chunk'la (sadece ilk 100 chunk test için)
    print("📖 Text işleniyor...")
    text = extract_text_from_pdf(pdf_path)
    
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_text(text, source_name=pdf_path.name)
    
    # Test için sadece ilk 100 chunk
    test_chunks = chunks[:100]
    print(f"✅ {len(test_chunks)} chunk hazır (test için)\n")
    
    # Embedder oluştur
    # DAHA İYİ MODEL (yavaş, 768-dim, %10-15 daha iyi)
    embedder = Embedder(model_name="all-mpnet-base-v2")
    print()
    
    # Embedding'leri oluştur
    embedded_docs = embedder.embed_documents(test_chunks)
    
    # İlk embedding'i göster
    print("\n" + "="*60)
    print("EMBEDDING ÖRNEĞİ")
    print("="*60)
    
    first_doc = embedded_docs[0]
    print(f"Text: {first_doc['text'][:100]}...")
    print(f"Embedding shape: {first_doc['embedding'].shape}")
    print(f"İlk 10 değer: {first_doc['embedding'][:10]}")
    
    # Similarity testi
    print("\n" + "="*60)
    print("SIMILARITY TEST")
    print("="*60)
    
    # Daha anlamlı test: Aynı bölümdeki chunk'ları karşılaştır
    # İlk 10 chunk muhtemelen aynı bölümde (Introduction)
    emb1 = embedded_docs[5]['embedding']  # 5. chunk
    emb2 = embedded_docs[6]['embedding']  # 6. chunk (hemen yan yana)
    emb3 = embedded_docs[70]['embedding'] # 70. chunk (çok uzak)

    sim_close = embedder.compute_similarity(emb1, emb2)
    sim_far = embedder.compute_similarity(emb1, emb3)

    print(f"Chunk 5 <-> Chunk 6 similarity: {sim_close:.4f} (yan yana chunk'lar)")
    print(f"Chunk 5 <-> Chunk 70 similarity: {sim_far:.4f} (uzak chunk'lar)")
    
    sim_close = embedder.compute_similarity(emb1, emb2)
    sim_far = embedder.compute_similarity(emb1, emb3)
    
    print(f"Chunk 0 <-> Chunk 1 similarity: {sim_close:.4f} (yakın chunk'lar)")
    print(f"Chunk 0 <-> Chunk 50 similarity: {sim_far:.4f} (uzak chunk'lar)")
    print("\n💡 Yakın chunk'lar daha yüksek similarity'e sahip olmalı!")
    
    print("\n✅ Embedding testi başarılı!")


if __name__ == "__main__":
    main()