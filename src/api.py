"""
FastAPI REST API - Mobil test için
"""
from typing import Optional, List, Dict
import time

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel, Field

import uvicorn  # type: ignore

from config import config
from rag_pipeline_hybrid import HybridRAGPipeline


# ============================================================
# MODELLER (Request/Response)
# ============================================================

class QueryRequest(BaseModel):
    """Soru sorma request"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Kullanıcının sorusu",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Retrieval için chunk sayısı",
    )
    use_web: Optional[bool] = Field(
        default=True,
        description="Web fallback kullanıldı mı?",
    )


class Source(BaseModel):
    """Kaynak bilgisi"""
    source: str
    chunk_id: int
    text_preview: str
    similarity: Optional[float] = None


class QueryResponse(BaseModel):
    """Soru cevap response"""
    success: bool
    answer: str
    confidence: float
    method_used: str  # "llama" veya "claude+web"
    sources: List[Source]
    web_enhanced: bool
    web_sources: Optional[List[Dict[str, str]]] = None
    response_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    ollama_status: bool
    claude_status: bool
    vector_db_count: int
    pdf_count: int
    pipeline_ready: bool


class ErrorResponse(BaseModel):
    """Hata response"""
    success: bool = False
    error: str
    error_type: str


# ============================================================
# API INITIALIZATION
# ============================================================

app = FastAPI(
    title="DnD RAG API",
    description="D&D 5e kuralları için RAG-based chatbot API",
    version="1.0.0",
    docs_url="/docs",   # Swagger UI
    redoc_url="/redoc", # ReDoc
)

# CORS (Mobil için ÇOK ÖNEMLİ ÖMER!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline (lazy init)
rag_pipeline: Optional[HybridRAGPipeline] = None
pipeline_ready: bool = False


# ============================================================
# LIFECYCLE EVENTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başladığında sadece config kontrolü yap.
    Ağır olan RAG pipeline kurulumu ilk /query isteğinde yapılacak (lazy init).
    """
    print(" API başlatılıyor (lazy RAG init)...")
    try:
        config.validate()
        print("✅ Config OK. RAG pipeline ilk istek geldiğinde oluşturulacak.")
        print(f"ℹ️ USE_OLLAMA = {config.USE_OLLAMA}")
    except Exception as e:
        print(f"❌ Config hatası: {e}")
        # Config bozuksa server hiç ayağa kalkmasın
        raise


# ============================================================
# ENDPOINTLER
# ============================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "message": "DnD RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Sistem sağlık kontrolü

    Returns:
        - Ollama durumu (USE_OLLAMA true ise kontrol edilir)
        - Claude durumu
        - Vector DB document sayısı (pipeline yüklüyse)
        - PDF sayısı
    """
    import requests
    from anthropic import Anthropic
    from pdf_processor import list_pdfs

    # Ollama kontrol (isteğe bağlı)
    ollama_ok = False
    if getattr(config, "USE_OLLAMA", False):
        try:
            r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

    # Claude kontrol
    claude_ok = False
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        claude_ok = True
    except Exception:
        claude_ok = False

    # Vector DB
    vector_count = 0
    if rag_pipeline is not None:
        try:
            vector_count = rag_pipeline.vector_db.collection.count()
        except Exception:
            vector_count = 0

    # PDF count
    try:
        pdf_count = len(list_pdfs())
    except Exception:
        pdf_count = 0

    status = "healthy" if (claude_ok and vector_count > 0) else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        ollama_status=ollama_ok,
        claude_status=claude_ok,
        vector_db_count=vector_count,
        pdf_count=pdf_count,
        pipeline_ready=pipeline_ready,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def ask_question(request: QueryRequest):
    """
    RAG sistemine soru sor

    ### Örnek Request:
    ```json
    {
        "question": "What are ability scores?",
        "top_k": 5,
        "use_web": true
    }
    ```
    """
    global rag_pipeline, pipeline_ready

    # İlk istek geldiğinde RAG pipeline'ı oluştur (lazy init)
    if rag_pipeline is None:
        try:
            print("⚙️  İlk istek geldi, RAG pipeline oluşturuluyor...")
            start_init = time.time()
            rag_pipeline = HybridRAGPipeline()
            pipeline_ready = True
            print(f"✅ RAG pipeline hazır. Süre: {time.time() - start_init:.2f} sn")
        except Exception as e:
            pipeline_ready = False
            print(f"❌ RAG pipeline oluşturulamadı: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline init hatası: {str(e)}",
            )

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline henüz hazır değil",
        )

    try:
        # Timing
        start_time = time.time()

        # RAG query
        result = rag_pipeline.query(
            request.question,
            top_k=request.top_k,
        )

        elapsed = time.time() - start_time

        # Response formatla
        return QueryResponse(
            success=True,
            answer=result["answer"],
            confidence=result["confidence"],
            method_used=result["method_used"],
            sources=[
                Source(
                    source=s["source"],
                    chunk_id=s["chunk_id"],
                    text_preview=s["text_preview"],
                    similarity=s.get("similarity"),
                )
                for s in result["sources"]
            ],
            web_enhanced=result.get("web_enhanced", False),
            web_sources=result.get("web_sources"),
            response_time=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query işlenirken hata: {str(e)}",
        )


@app.get("/stats", tags=["General"])
async def get_stats():
    """Sistem istatistikleri"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline yüklenmedi")

    stats = rag_pipeline.vector_db.get_stats()

    return {
        "vector_db": stats,
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "top_k": config.TOP_K,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "ollama_model": config.OLLAMA_MODEL,
            "claude_model": config.CLAUDE_MODEL,
            "use_ollama": getattr(config, "USE_OLLAMA", False),
        },
    }


# ============================================================
# RUN (Local geliştirme için)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DnD RAG API Starting")
    print("=" * 60)
    print("📖 Docs:  http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run(
        "api:app",
        host="0.0.0.0",  # Tüm network interface'lerden erişilebilir
        port=8000,
        reload=True,     # Development için auto-reload
        log_level="info",
    )
