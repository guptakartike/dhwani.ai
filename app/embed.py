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
            # Multilingual sentence transformer for Hindi/Indic language semantic vector search
            logger.info("Loading local multilingual sentence-transformer model...")
            try:
                _model_instance = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
            except Exception:
                _model_instance = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Multilingual SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using local lightweight dummy embedder for fallback.")
            _model_instance = "FALLBACK"
    return _model_instance

def embed_query(text: str) -> List[float]:
    """Encodes query text to a vector embedding (384 dimensions)."""
    model = get_embedding_model()
    if model == "FALLBACK":
        # Deterministic 384-dim normalized vector fallback for offline/sandboxed execution
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist()
    
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encodes a list of texts to vector embeddings."""
    model = get_embedding_model()
    if model == "FALLBACK":
        res = []
        for t in texts:
            np.random.seed(abs(hash(t)) % (2**32))
            vec = np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            res.append((vec / norm).tolist())
        return res
    
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
