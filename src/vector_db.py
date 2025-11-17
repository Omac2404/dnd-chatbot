"""
Vector Database modülü
ChromaDB ile embedding'leri saklar ve arar
"""

import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from typing import List, Dict
import numpy as np # type: ignore
from config import config


class VectorDB:
    """ChromaDB wrapper sınıfı"""
    
    def __init__(self, collection_name: str = "dnd_knowledge"):
        """
        Args:
            collection_name: Koleksiyon adı (veritabanı tablosu gibi)
        """
        # ChromaDB client oluştur (persistent storage)
        self.client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_DIR)
        )
        
        # Collection oluştur veya mevcut olanı getir
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "D&D 5e knowledge base"}
        )
        
        print(f"✅ ChromaDB hazır: {collection_name}")
        print(f"📊 Mevcut document sayısı: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict]):
        """
        Embedding'li document'ları database'e ekle
        
        Args:
            documents: embedder'dan gelen documents (embedding field'ı olmalı)
        """
        if not documents:
            print("⚠️ Eklenecek document yok!")
            return
        
        # ChromaDB formatına çevir
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc['text'] for doc in documents]
        embeddings = [doc['embedding'].tolist() for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        print(f"💾 {len(documents)} document database'e ekleniyor...")
        
        # Batch olarak ekle (ChromaDB 5000'lik batch'leri sever)
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            print(f"   ✅ {end_idx}/{len(documents)} eklendi")
        
        print(f"✅ Toplam {self.collection.count()} document database'de")
    
    def search(self, query_text: str, n_results: int = 5, query_embedding: np.ndarray = None) -> List[Dict]:
        """
        Query text'ine benzer document'ları ara
        
        Args:
            query_text: Aranacak text
            n_results: Kaç sonuç döndürülsün (top-k)
            query_embedding: Önceden hazırlanmış query embedding (opsiyonel)
            
        Returns:
            En benzer document'ların listesi
        """
        # Eğer embedding verilmişse onu kullan
        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
        else:
            # Text ile ara (ChromaDB kendi embedding'ini kullanır)
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Distance'ı similarity'ye çevir
            distance = results['distances'][0][i] if 'distances' in results else None
            similarity = None
            if distance is not None:
                # ChromaDB L2 distance kullanır, 0-2 arası normalize edelim
                similarity = max(0, 1 - (distance / 2))
            
            doc = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'similarity': similarity
            }
            formatted_results.append(doc)
        
        return formatted_results
    
    def clear(self):
        """Database'i temizle"""
        self.client.delete_collection(self.collection.name)
        print(f"🗑️ Collection '{self.collection.name}' silindi")
    
    def get_stats(self) -> Dict:
        """Database istatistikleri"""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "storage_path": str(config.VECTOR_DB_DIR)
        }


def main():
    """Test scripti"""
    from text_chunker import TextChunker
    from embedder import Embedder
    from pdf_processor import extract_text_from_pdf, list_pdfs
    
    print("="*60)
    print("VECTOR DATABASE TEST")
    print("="*60 + "\n")
    
    # PDF'leri listele
    pdfs = list_pdfs()
    
    if not pdfs:
        print("❌ PDF bulunamadı!")
        return
    
    pdf_path = pdfs[0]
    print(f"📄 Test PDF: {pdf_path.name}\n")
    
    # Text işle
    print("📖 Text işleniyor...")
    text = extract_text_from_pdf(pdf_path)
    
    chunker = TextChunker()
    chunks = chunker.chunk_text(text, source_name=pdf_path.name)
    
    # Gerçek içerik için 100-300 arası chunk'lar (ilk 100 genelde Contents/Intro)
    test_chunks = chunks  # TÜM CHUNK'LAR (~2900)
    print(f"✅ {len(test_chunks)} chunk hazır (TÜM PDF)\n")
    
    # DOĞRU MODEL ile embedding (all-mpnet-base-v2, 768-dim)
    print("🔤 Embedding modeli yükleniyor...")
    embedder = Embedder(model_name="all-mpnet-base-v2")
    embedded_docs = embedder.embed_documents(test_chunks)
    print()
    
    # Vector DB'ye kaydet
    db = VectorDB(collection_name="dnd_test")
    
    # Önce temizle (test için)
    if db.collection.count() > 0:
        print("🗑️ Eski data temizleniyor...")
        db.clear()
        db = VectorDB(collection_name="dnd_test")
    
    db.add_documents(embedded_docs)
    
    # Test search
    print("\n" + "="*60)
    print("SEARCH TEST")
    print("="*60)
    
    test_queries = [
        "What are ability scores in D&D?",
        "How do I calculate armor class?",
        "What is a saving throw?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        # Query'yi embedding'e çevir (aynı model ile!)
        query_embedding = embedder.embed_text(query)
        
        # Embedding ile ara
        results = db.search(query, n_results=3, query_embedding=query_embedding)
        
        print(f"📊 En alakalı {len(results)} sonuç:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Kaynak: {result['metadata']['source']}")
            print(f"   Chunk: {result['metadata']['chunk_id']}")
            print(f"   Text: {result['text'][:150]}...")
            if result['distance'] is not None:
                print(f"   Distance: {result['distance']:.4f} (düşük = iyi)")
            if result['similarity'] is not None:
                print(f"   Similarity: {result['similarity']:.4f} (yüksek = iyi)")
    
    # Stats
    print("\n" + "="*60)
    print("DATABASE STATS")
    print("="*60)
    
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Vector database testi başarılı!")


if __name__ == "__main__":
    main()