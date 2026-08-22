import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None
    VectorParams = Distance = PointStruct = None

from dotenv import load_dotenv
from app.schemas import RetrievedChunk, RetrievalResponse
from app.embed import embed_query, embed_texts

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTIONS = {
    "passage": "msmarco_passage",
    "sentence_window": "msmarco_sentence_window",
    "semantic": "msmarco_semantic"
}

_qdrant_client: Any = None
_fallback_store: Dict[str, List[Dict[str, Any]]] = {
    "passage": [],
    "sentence_window": [],
    "semantic": []
}

def get_qdrant_client() -> Any:
    global _qdrant_client
    if not HAS_QDRANT:
        logger.info("qdrant_client package not loaded. Using local in-memory vector store.")
        _ensure_fallback_seeded()
        return "FALLBACK_STORE"

    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY", "")
        try:
            if qdrant_url.startswith("http") and "cloud.qdrant.io" in qdrant_url and qdrant_key:
                _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=3.0)
                _qdrant_client.get_collections() # test connection
                logger.info(f"Connected to Qdrant Cloud at {qdrant_url}")
                _seed_qdrant_memory(_qdrant_client)
            else:
                _qdrant_client = QdrantClient(":memory:")
                logger.info("Initialized in-memory Qdrant client")
                _seed_qdrant_memory(_qdrant_client)
        except Exception as e:
            logger.warning(f"Could not connect to remote Qdrant ({e}). Falling back to in-memory Qdrant Client.")
            try:
                _qdrant_client = QdrantClient(":memory:")
                _seed_qdrant_memory(_qdrant_client)
            except Exception:
                _qdrant_client = "FALLBACK_STORE"
                _ensure_fallback_seeded()
    return _qdrant_client

def _seed_qdrant_memory(client: Any):
    """Auto-seeds MSMARCO dataset chunks into in-memory Qdrant client."""
    try:
        from data.ingest import load_dataset_samples, passage_chunking, sentence_window_chunking, semantic_chunking
        samples = load_dataset_samples(limit=10, use_remote=False)
        all_chunks = []
        for sample in samples:
            qid = sample["query_id"]
            passages = sample["passages"]
            all_chunks.extend(passage_chunking(passages, qid))
            all_chunks.extend(sentence_window_chunking(passages, qid))
            all_chunks.extend(semantic_chunking(passages, qid))
        index_chunks(all_chunks, client)
        logger.info(f"Auto-seeded {len(all_chunks)} chunks into in-memory Qdrant vector store.")
    except Exception as e:
        logger.warning(f"Auto-seeding in-memory Qdrant failed: {e}")

def _ensure_fallback_seeded():
    """Auto-seeds fallback store if empty."""
    if not any(_fallback_store.values()):
        try:
            from data.ingest import load_dataset_samples, passage_chunking, sentence_window_chunking, semantic_chunking
            samples = load_dataset_samples(limit=10, use_remote=False)
            all_chunks = []
            for sample in samples:
                qid = sample["query_id"]
                passages = sample["passages"]
                all_chunks.extend(passage_chunking(passages, qid))
                all_chunks.extend(sentence_window_chunking(passages, qid))
                all_chunks.extend(semantic_chunking(passages, qid))
            index_chunks(all_chunks, "FALLBACK_STORE")
        except Exception as e:
            logger.warning(f"Auto-seeding fallback store failed: {e}")

def init_qdrant_collections(client: Any, vector_dim: int = 384):
    """Ensures Qdrant collections for all 3 chunking strategies exist."""
    if client == "FALLBACK_STORE" or not HAS_QDRANT:
        return
    existing = [c.name for c in client.get_collections().collections]
    for strategy, name in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )
            logger.info(f"Created collection '{name}' for strategy '{strategy}'")

def index_chunks(chunks: List[Dict[str, Any]], client: Optional[Any] = None):
    """Embeds and indexes chunks into their respective Qdrant collections or fallback store."""
    if client is None:
        client = get_qdrant_client()
    
    if client == "FALLBACK_STORE" or not HAS_QDRANT:
        for c in chunks:
            strat = c.get("strategy", "passage")
            vec = embed_query(c["text"])
            _fallback_store.setdefault(strat, []).append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "vector": np.array(vec, dtype=np.float32),
                "strategy": strat,
                "query_id": c.get("query_id", ""),
                "is_selected": c.get("is_selected", 0),
                "position": c.get("position", 0)
            })
        logger.info(f"Indexed {len(chunks)} chunks into local fallback store.")
        return

    init_qdrant_collections(client)
    
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        strat = c.get("strategy", "passage")
        grouped.setdefault(strat, []).append(c)
        
    for strat, items in grouped.items():
        coll_name = COLLECTIONS.get(strat, COLLECTIONS["passage"])
        texts = [item["text"] for item in items]
        vectors = embed_texts(texts)
        
        points = []
        for idx, (item, vec) in enumerate(zip(items, vectors)):
            points.append(
                PointStruct(
                    id=abs(hash(item["chunk_id"])) % (2**63 - 1),
                    vector=vec,
                    payload={
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "strategy": strat,
                        "query_id": item.get("query_id", ""),
                        "is_selected": item.get("is_selected", 0),
                        "position": item.get("position", 0)
                    }
                )
            )
        client.upsert(collection_name=coll_name, points=points)
        logger.info(f"Indexed {len(points)} points into '{coll_name}' ({strat})")

def search_retrieval(query: str, top_k: int = 5, client: Optional[Any] = None) -> RetrievalResponse:
    """Searches across all 3 chunking collections and returns ranked results with strategy attribution."""
    if client is None:
        client = get_qdrant_client()
        
    query_vec = np.array(embed_query(query), dtype=np.float32)
    all_chunks: List[RetrievedChunk] = []

    if client == "FALLBACK_STORE" or not HAS_QDRANT:
        for strat, items in _fallback_store.items():
            for item in items:
                v = item["vector"]
                score = float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-9))
                all_chunks.append(
                    RetrievedChunk(
                        chunk_id=item["chunk_id"],
                        text=item["text"],
                        score=score,
                        strategy=strat,
                        query_id=item.get("query_id", ""),
                        is_selected=item.get("is_selected", 0),
                        position=item.get("position", 0)
                    )
                )
    else:
        for strat, coll_name in COLLECTIONS.items():
            try:
                hits = []
                if hasattr(client, "query_points"):
                    res = client.query_points(
                        collection_name=coll_name,
                        query=query_vec.tolist(),
                        limit=top_k
                    )
                    hits = res.points
                elif hasattr(client, "search"):
                    hits = client.search(
                        collection_name=coll_name,
                        query_vector=query_vec.tolist(),
                        limit=top_k
                    )
                
                for hit in hits:
                    all_chunks.append(
                        RetrievedChunk(
                            chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                            text=hit.payload.get("text", ""),
                            score=float(hit.score),
                            strategy=strat,
                            query_id=hit.payload.get("query_id", ""),
                            is_selected=hit.payload.get("is_selected", 0),
                            position=hit.payload.get("position", 0)
                        )
                    )
            except Exception as e:
                logger.warning(f"Error searching Qdrant collection '{coll_name}': {e}")
            
    # Sort all retrieved chunks by similarity score descending
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    top_chunks = all_chunks[:top_k]
    
    top_score = top_chunks[0].score if top_chunks else 0.0
    best_strategy = top_chunks[0].strategy if top_chunks else None
    
    return RetrievalResponse(
        chunks=top_chunks,
        top_score=top_score,
        best_strategy=best_strategy
    )
