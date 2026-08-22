import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sample Hindi MSMARCO passages as fallback dataset if HuggingFace download is unavailable/offline
SAMPLE_MSMARCO_HINDI = [
    {
        "query_id": "hi_q1",
        "query": "भारत की राजधानी क्या है?",
        "answers": ["भारत की राजधानी नई दिल्ली है।"],
        "passages": [
            {
                "is_selected": 1,
                "passage_text": "नई दिल्ली भारत की राजधानी और केंद्र शासित प्रदेश है। यह भारत सरकार के तीन अंगों: कार्यपालिका, विधायिका और न्यायपालिका का केंद्र है।"
            },
            {
                "is_selected": 0,
                "passage_text": "मुंबई भारत के महाराष्ट्र राज्य की राजधानी है। इसे भारत की वित्तीय राजधानी भी कहा जाता है।"
            }
        ]
    },
    {
        "query_id": "hi_q2",
        "query": "सूर्य ग्रहण कैसे होता है?",
        "answers": ["जब चंद्रमा पृथ्वी और सूर्य के बीच आ जाता है, तो सूर्य ग्रहण होता है।"],
        "passages": [
            {
                "is_selected": 1,
                "passage_text": "सूर्य ग्रहण तब होता है जब चंद्रमा पृथ्वी और सूर्य के बीच आ जाता है, जिससे सूर्य का प्रकाश आंशिक या पूर्ण रूप से अवरुद्ध हो जाता है। यह अमावस्या के दिन होता है।"
            },
            {
                "is_selected": 0,
                "passage_text": "चंद्र ग्रहण तब होता है जब पृथ्वी सूर्य और चंद्रमा के बीच आ जाती है।"
            }
        ]
    },
    {
        "query_id": "hi_q3",
        "query": "प्रकाश संश्लेषण की प्रक्रिया क्या है?",
        "answers": ["पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके अपना भोजन बनाते हैं।"],
        "passages": [
            {
                "is_selected": 1,
                "passage_text": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश, पानी और कार्बन डाइऑक्साइड का उपयोग करके ग्लूकोज और ऑक्सीजन का निर्माण करते हैं।"
            },
            {
                "is_selected": 0,
                "passage_text": "श्वसन प्रक्रिया में पौधे ऑक्सीजन लेते हैं और कार्बन डाइऑक्साइड छोड़ते हैं।"
            }
        ]
    }
]

def split_sentences_hindi(text: str) -> List[str]:
    """Splits Hindi text into sentences based on '।', '.', '?', '!'."""
    delimiters = ['।', '.', '?', '!']
    sentences = [text]
    for d in delimiters:
        temp = []
        for s in sentences:
            temp.extend(s.split(d))
        sentences = temp
    return [s.strip() for s in sentences if s.strip()]

def passage_chunking(passages: List[Dict[str, Any]], query_id: str) -> List[Dict[str, Any]]:
    """Strategy 1: Native passage-level chunking."""
    chunks = []
    for idx, p in enumerate(passages):
        chunks.append({
            "chunk_id": f"{query_id}_p_{idx}",
            "text": p["passage_text"],
            "strategy": "passage",
            "query_id": query_id,
            "is_selected": p.get("is_selected", 0),
            "position": idx
        })
    return chunks

def sentence_window_chunking(passages: List[Dict[str, Any]], query_id: str, window_size: int = 1) -> List[Dict[str, Any]]:
    """Strategy 2: Sentence-window chunking (context window of +/- window_size sentences)."""
    chunks = []
    chunk_count = 0
    for p_idx, p in enumerate(passages):
        sentences = split_sentences_hindi(p["passage_text"])
        for i, s in enumerate(sentences):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window_text = " ".join(sentences[start:end])
            chunks.append({
                "chunk_id": f"{query_id}_sw_{p_idx}_{i}",
                "text": window_text,
                "target_sentence": s,
                "strategy": "sentence_window",
                "query_id": query_id,
                "is_selected": p.get("is_selected", 0),
                "position": chunk_count
            })
            chunk_count += 1
    return chunks

def semantic_chunking(passages: List[Dict[str, Any]], query_id: str, embedder=None) -> List[Dict[str, Any]]:
    """Strategy 3: Semantic chunking by grouping sentences with high similarity."""
    chunks = []
    chunk_count = 0
    for p_idx, p in enumerate(passages):
        sentences = split_sentences_hindi(p["passage_text"])
        if not sentences:
            continue
        
        if embedder and len(sentences) > 1:
            try:
                embeddings = embedder.encode(sentences)
                import numpy as np
                current_chunk = [sentences[0]]
                
                for i in range(1, len(sentences)):
                    sim = np.dot(embeddings[i-1], embeddings[i]) / (
                        np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-9
                    )
                    if sim >= 0.7:  # Merge if semantically close
                        current_chunk.append(sentences[i])
                    else:
                        chunks.append({
                            "chunk_id": f"{query_id}_sem_{p_idx}_{chunk_count}",
                            "text": " ".join(current_chunk),
                            "strategy": "semantic",
                            "query_id": query_id,
                            "is_selected": p.get("is_selected", 0),
                            "position": chunk_count
                        })
                        chunk_count += 1
                        current_chunk = [sentences[i]]
                
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{query_id}_sem_{p_idx}_{chunk_count}",
                        "text": " ".join(current_chunk),
                        "strategy": "semantic",
                        "query_id": query_id,
                        "is_selected": p.get("is_selected", 0),
                        "position": chunk_count
                    })
                    chunk_count += 1
                continue
            except Exception as e:
                logger.warning(f"Semantic chunking embedding fallback: {e}")
        
        # Fallback if embedder not ready: combine pairs of sentences
        for i in range(0, len(sentences), 2):
            combined = " ".join(sentences[i:i+2])
            chunks.append({
                "chunk_id": f"{query_id}_sem_{p_idx}_{chunk_count}",
                "text": combined,
                "strategy": "semantic",
                "query_id": query_id,
                "is_selected": p.get("is_selected", 0),
                "position": chunk_count
            })
            chunk_count += 1
            
    return chunks

def load_dataset_samples(limit: int = 50, use_remote: bool = False) -> List[Dict[str, Any]]:
    """Loads MSMARCO-XI Hindi dataset with fallback to local sample data."""
    if not use_remote:
        return SAMPLE_MSMARCO_HINDI
    try:
        from datasets import load_dataset
        logger.info("Attempting to load ai4bharat/MSMARCO-XI Hindi from HuggingFace...")
        ds = load_dataset("ai4bharat/MSMARCO-XI", data_files="hi/*", split=f"train[:{limit}]")
        records = []
        for row in ds:
            records.append({
                "query_id": str(row.get("query_id", len(records))),
                "query": row.get("query", ""),
                "answers": row.get("answers", []),
                "passages": row.get("passages", [])
            })
        logger.info(f"Successfully loaded {len(records)} items from HuggingFace.")
        return records
    except Exception as e:
        logger.warning(f"Could not load dataset from HuggingFace ({e}). Using sample dataset fallback.")
        return SAMPLE_MSMARCO_HINDI

if __name__ == "__main__":
    data = load_dataset_samples(limit=5)
    print(f"Loaded {len(data)} items.")
    sample = data[0]
    print("Sample Query:", sample["query"])
    print("Sample Passages Count:", len(sample["passages"]))
    print("Sample Answers:", sample["answers"])
    
    p_chunks = passage_chunking(sample["passages"], sample["query_id"])
    sw_chunks = sentence_window_chunking(sample["passages"], sample["query_id"])
    sem_chunks = semantic_chunking(sample["passages"], sample["query_id"])
    
    print(f"Chunks generated -> Passage: {len(p_chunks)}, Sentence-Window: {len(sw_chunks)}, Semantic: {len(sem_chunks)}")
