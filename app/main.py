import os
import sys
import time
import logging
from typing import Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.schemas import (
    RAGResponse, RetrievalResponse, GenerationResult, RefusalReason, TranscriptResult
)
from app.retrieval import search_retrieval
from app.generate import GroqGenerator
from app.guardrails import GuardrailGate
from app.stt import SarvamSTTService, transcribe_text_mock

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice-Enabled RAG Pipeline",
    description="High-performance Hindi Voice RAG pipeline with multi-strategy retrieval, sub-200ms target, and hard guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vesper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vesper_frontend")

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

generator = GroqGenerator()
guardrail = GuardrailGate(confidence_threshold=0.45)
stt_service = SarvamSTTService()

@app.on_event("startup")
def warmup_services():
    """Pre-warms vector DB, sentence transformer MPS model, and metal shaders at startup."""
    logger.info("Pre-warming Qdrant vector database and embedding model...")
    try:
        from app.retrieval import get_qdrant_client
        from app.embed import get_embedding_model, embed_query
        get_embedding_model()
        embed_query("warmup query")
        get_qdrant_client()
        logger.info("Server pre-warming complete. Sub-200ms query latency ready.")
    except Exception as e:
        logger.warning(f"Startup pre-warming warning: {e}")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    confidence_threshold: float = 0.45
    target_lang: str = "hi"

@app.get("/")
def read_root():
    vesper_index = os.path.join(vesper_dir, "index.html")
    if os.path.exists(vesper_index):
        return FileResponse(vesper_index)
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/architecture")
def read_architecture():
    arch_file = os.path.join(vesper_dir, "architecture.html")
    if os.path.exists(arch_file):
        return FileResponse(arch_file)
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/overview")
def read_overview():
    overview_file = os.path.join(vesper_dir, "overview.html")
    if os.path.exists(overview_file):
        return FileResponse(overview_file)
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/favicon.ico")
@app.get("/favicon.svg")
def get_favicon():
    favicon_path = os.path.join(static_dir, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/query", response_model=RAGResponse)
def execute_rag_query(request: QueryRequest) -> RAGResponse:
    """Executes post-transcription RAG pipeline (Retrieval -> Guardrail Gate -> Generation -> Output Validation)."""
    start_total = time.perf_counter()
    metrics: Dict[str, float] = {}

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # 1. Multi-Strategy Retrieval
    t0 = time.perf_counter()
    retrieval_resp: RetrievalResponse = search_retrieval(query, top_k=request.top_k)
    metrics["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Pre-Generation Guardrail Gate
    t0 = time.perf_counter()
    passed_pre, refusal_pre = guardrail.evaluate_pre_generation(query, retrieval_resp)
    metrics["guardrail_pre_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not passed_pre or refusal_pre:
        metrics["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
        return RAGResponse(
            success=False,
            query=query,
            answer=None,
            refusal=refusal_pre,
            sources=[],
            metrics=metrics
        )

    # 3. Grounded LLM Generation
    t0 = time.perf_counter()
    gen_result: GenerationResult = generator.generate_answer(query, retrieval_resp.chunks, target_lang=request.target_lang)
    metrics["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 4. Post-Generation Guardrail Gate
    t0 = time.perf_counter()
    passed_post, refusal_post = guardrail.evaluate_post_generation(query, gen_result)
    metrics["guardrail_post_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not passed_post or refusal_post:
        metrics["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
        return RAGResponse(
            success=False,
            query=query,
            answer=None,
            refusal=refusal_post,
            sources=[],
            metrics=metrics
        )

    metrics["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
    return RAGResponse(
        success=True,
        query=query,
        answer=gen_result.answer,
        refusal=None,
        sources=gen_result.sources,
        metrics=metrics
    )

@app.websocket("/ws/rag")
async def websocket_rag_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming STT -> RAG pipeline."""
    await websocket.accept()
    logger.info("WebSocket connection established for /ws/rag")

    async def audio_generator():
        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    break
                yield data
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")

    try:
        async for stt_result in stt_service.transcribe_stream(audio_generator()):
            # Send STT updates back to client
            await websocket.send_json({
                "type": "stt",
                "text": stt_result.text,
                "is_final": stt_result.is_final
            })
            
            # When final transcript is emitted, run RAG pipeline
            if stt_result.is_final and stt_result.text.strip():
                rag_req = QueryRequest(query=stt_result.text)
                rag_resp = execute_rag_query(rag_req)
                await websocket.send_json({
                    "type": "rag_result",
                    "payload": rag_resp.model_dump()
                })
    except Exception as e:
        logger.error(f"WebSocket pipeline error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
