"""
Tüm PDF'leri işleyip ChromaDB'ye yükle
Bu script bir kez çalıştırılmalı (database build yapar)
"""

from text_chunker import TextChunker
from embedder import Embedder
from vector_db import VectorDB
from pdf_processor import extract_text_from_pdf, list_pdfs


def build_knowledge_base():
    """PDF'lerden knowledge base oluştur"""
    
    print("="*60)
    print("KNOWLEDGE BASE BUILDER")
    print("="*60 + "\n")
    
    # PDF'leri listele
    pdfs = list_pdfs()
    
    if not pdfs:
        print("❌ PDF bulunamadı!")
        return
    
    print(f"📚 {len(pdfs)} PDF bulundu\n")
    
    # Tüm chunk'ları topla
    all_chunks = []
    
    for pdf_path in pdfs:
        print(f"📄 İşleniyor: {pdf_path.name}")
        
        # Text çıkar
        text = extract_text_from_pdf(pdf_path)
        print(f"   📖 {len(text):,} karakter çıkarıldı")
        
        # Chunk'la
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_text(text, source_name=pdf_path.name)
        print(f"   ✂️ {len(chunks)} chunk oluşturuldu")
        
        all_chunks.extend(chunks)
    
    print(f"\n✅ Toplam {len(all_chunks)} chunk hazır\n")
    
    # ÖNEMLİ: all-mpnet-base-v2 kullan (768-dim) mini model başarısız!!
    print("🔤 Embedding'ler oluşturuluyor...")
    embedder = Embedder(model_name="all-mpnet-base-v2")
    embedded_docs = embedder.embed_documents(all_chunks)
    
    # ChromaDB'ye kaydet
    print("\n💾 ChromaDB'ye kaydediliyor...")
    db = VectorDB(collection_name="dnd_knowledge")
    
    # Eski data varsa temizle
    if db.collection.count() > 0:
        print(f"🗑️ Eski {db.collection.count()} document temizleniyor...")
        db.clear()
        db = VectorDB(collection_name="dnd_knowledge")
    
    db.add_documents(embedded_docs)
    
    # Statlar
    print("\n" + "="*60)
    print("DATABASE STATS")
    print("="*60)
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Knowledge base başarıyla oluşturuldu!")
    print(f"📊 Toplam {db.collection.count()} document veritabanında")
    print("\n💡 Artık RAG pipeline'ını çalıştırabilirsiniz!")


if __name__ == "__main__":
    build_knowledge_base()