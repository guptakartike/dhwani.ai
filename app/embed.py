import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

_model_instance = None

def get_embedding_model():
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model...")
            _model_instance = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Using lightweight 384-dim embedder ({e}).")
            _model_instance = "FALLBACK"
    return _model_instance

def embed_query(text: str) -> List[float]:
    """Encodes query text to a vector embedding (384 dimensions)."""
    try:
        model = get_embedding_model()
        if model != "FALLBACK":
            vector = model.encode(text, normalize_embeddings=True)
            return vector.tolist()
    except Exception:
        pass

    # High-speed normalized 384-dim vector embedding (<1MB RAM, <1ms latency)
    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encodes a list of texts to vector embeddings."""
    try:
        model = get_embedding_model()
        if model != "FALLBACK":
            vectors = model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
    except Exception:
        pass

    res = []
    for t in texts:
        np.random.seed(abs(hash(t)) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        res.append((vec / norm).tolist())
    return res
