from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TranscriptResult(BaseModel):
    text: str = Field(..., description="Transcribed text from speech input")
    is_final: bool = Field(default=True, description="True if final transcript, False if partial stream")
    confidence: float = Field(default=1.0, description="Speech-to-text confidence score")

class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk content text")
    score: float = Field(..., description="Vector search similarity score")
    strategy: str = Field(..., description="Chunking strategy used: passage, sentence_window, or semantic")
    query_id: str = Field(..., description="Query ID association")
    is_selected: int = Field(default=0, description="Relevance flag from dataset")
    position: int = Field(default=0, description="Chunk position index")

class RetrievalResponse(BaseModel):
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Top-k retrieved chunks")
    top_score: float = Field(default=0.0, description="Highest similarity score among retrieved chunks")
    best_strategy: Optional[str] = Field(default=None, description="Strategy that yielded the highest scoring chunk")

class GenerationResult(BaseModel):
    answer: str = Field(..., description="LLM generated answer text")
    grounded: bool = Field(..., description="True if answer is directly derived from retrieved context")
    sources: List[str] = Field(default_factory=list, description="Source chunk IDs used in generating answer")

class RefusalReason(BaseModel):
    code: str = Field(..., description="Refusal code, e.g. LOW_RETRIEVAL_CONFIDENCE, UNGROUNDED_OUTPUT, HARMFUL_QUERY")
    message: str = Field(..., description="User-facing refusal explanation")
    score: Optional[float] = Field(default=None, description="Score that triggered refusal")

class RAGResponse(BaseModel):
    success: bool = Field(..., description="True if answer generated, False if refused or error")
    query: str = Field(..., description="Input query text")
    answer: Optional[str] = Field(default=None, description="Final generated answer")
    refusal: Optional[RefusalReason] = Field(default=None, description="Refusal details if request was rejected")
    sources: List[str] = Field(default_factory=list, description="List of source chunk IDs")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Latency breakdown in milliseconds (STT, Embed, Retrieval, Guardrail, Gen)")
