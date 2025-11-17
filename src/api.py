"""
FastAPI REST API - Mobil test için
"""
from fastapi import FastAPI, HTTPException, UploadFile, File # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn # type: ignore
import time
from pathlib import Path

# Mevcut modülleriniz
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

# Global RAG pipeline
rag_pipeline: Optional[HybridRAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında RAG pipeline'ı yükle"""
    global rag_pipeline
    print("🚀 API başlatılıyor...")

    try:
        # RAG pipeline'ı yükle
        rag_pipeline = HybridRAGPipeline()
        print("✅ RAG Pipeline yüklendi")
    except Exception as e:
        print(f"❌ RAG Pipeline yüklenemedi: {e}")
        # Production'da burada raise yapmalısınız
        raise


# ============================================================
# ENDPOINTLER !!
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
        - Ollama durumu
        - Claude durumu
        - Vector DB document sayısı
        - PDF sayısı
    """
    import requests
    from anthropic import Anthropic
    from pdf_processor import list_pdfs

    # Ollama kontrol
    ollama_ok = False
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except:
        pass

    # Claude kontrol
    claude_ok = False
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        # Basit bir istek
        client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        claude_ok = True
    except:
        pass

    # Vector DB
    vector_count = 0
    try:
        vector_count = rag_pipeline.vector_db.collection.count()
    except:
        pass

    # PDF count
    pdf_count = len(list_pdfs())

    return HealthResponse(
        status="healthy" if (ollama_ok and vector_count > 0) else "degraded",
        version="1.0.0",
        ollama_status=ollama_ok,
        claude_status=claude_ok,
        vector_db_count=vector_count,
        pdf_count=pdf_count,
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
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG Pipeline henüz yüklenmedi",
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
        },
    }


# ============================================================
# RUN
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
