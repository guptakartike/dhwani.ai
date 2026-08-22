import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingest import load_dataset_samples, passage_chunking, sentence_window_chunking, semantic_chunking
from app.retrieval import index_chunks, search_retrieval, get_qdrant_client
from app.main import execute_rag_query, QueryRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_OUTPUT_FILE = "benchmark_results.json"

# Expanded test queries (30 queries) for latency benchmarking
BENCHMARK_QUERIES = [
    "भारत की राजधानी क्या है?",
    "सूर्य ग्रहण कैसे होता है?",
    "प्रकाश संश्लेषण की प्रक्रिया क्या है?",
    "मुंबई को भारत की कौन सी राजधानी कहा जाता है?",
    "चंद्र ग्रहण कब होता है?",
    "हरे पौधे कौन सी गैस छोड़ते हैं?",
    "अमावस्या के दिन कौन सा ग्रहण होता है?",
    "भारत सरकार के तीन मुख्य अंग कौन से हैं?",
    "महाराष्ट्र राज्य की राजधानी क्या है?",
    "प्रकाश संश्लेषण में कौन से घटक आवश्यक हैं?",
    "क्या चंद्रमा सूर्य और पृथ्वी के बीच आता है?",
    "पौधे भोजन बनाने के लिए किसका उपयोग करते हैं?",
    "श्वसन प्रक्रिया में पौधे क्या करते हैं?",
    "भारत की वित्तीय राजधानी कौन सी है?",
    "कार्यपालिका और विधायिका कहां स्थित हैं?",
    "ग्लूकोज और ऑक्सीजन का निर्माण कैसे होता है?",
    "मंगल ग्रह पर जीवन की संभावना क्या है?",
    "कंप्यूटर का आविष्कार किसने किया?",
    "ताजमहल किस शहर में स्थित है?",
    "महात्मा गांधी का जन्म कब हुआ था?",
    "क्रिकेट विश्व कप 2023 किसने जीता?",
    "हिमालय पर्वत की सबसे ऊंची चोटी कौन सी है?",
    "गंगा नदी कहां से निकलती है?",
    "भारत का संविधान कब लागू हुआ था?",
    "एआई और मशीन लर्निंग में क्या अंतर है?",
    "सौर मंडल में कुल कितने ग्रह हैं?",
    "पानी का रासायनिक सूत्र क्या है?",
    "अंतरिक्ष में जाने वाले पहले भारतीय कौन थे?",
    "पायथन प्रोग्रामिंग भाषा के लेखक कौन हैं?",
    "योग दिवस कब मनाया जाता है?"
]

def run_benchmark(num_queries: int = 30) -> Dict[str, Any]:
    """Runs latency benchmark suite across post-transcription pipeline."""
    logger.info("Initializing Qdrant index with MSMARCO dataset chunks...")
    
    data_samples = load_dataset_samples(limit=10)
    all_chunks = []
    for sample in data_samples:
        qid = sample["query_id"]
        passages = sample["passages"]
        all_chunks.extend(passage_chunking(passages, qid))
        all_chunks.extend(sentence_window_chunking(passages, qid))
        all_chunks.extend(semantic_chunking(passages, qid))
        
    client = get_qdrant_client()
    index_chunks(all_chunks, client)
    logger.info(f"Indexed {len(all_chunks)} chunks for benchmark testing.")

    queries = BENCHMARK_QUERIES[:num_queries]
    latencies: List[float] = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    detailed_results = []

    logger.info(f"Running benchmark on {len(queries)} test queries...")

    for idx, q in enumerate(queries):
        start_time = time.perf_counter()
        req = QueryRequest(query=q, top_k=3)
        res = execute_rag_query(req)
        total_lat = (time.perf_counter() - start_time) * 1000

        latencies.append(total_lat)
        retrieval_latencies.append(res.metrics.get("retrieval_ms", 0.0))
        generation_latencies.append(res.metrics.get("generation_ms", 0.0))

        detailed_results.append({
            "query_index": idx + 1,
            "query": q,
            "success": res.success,
            "total_ms": round(total_lat, 2),
            "metrics": res.metrics,
            "refused": res.refusal is not None,
            "refusal_reason": res.refusal.code if res.refusal else None
        })

    p50 = float(np.percentile(latencies, 50))
    p70 = float(np.percentile(latencies, 70))
    p100 = float(np.percentile(latencies, 100))
    mean_lat = float(np.mean(latencies))

    summary = {
        "total_queries_tested": len(queries),
        "target_latency_ms": 200.0,
        "latency_percentiles_ms": {
            "p50": round(p50, 2),
            "p70": round(p70, 2),
            "p100": round(p100, 2),
            "mean": round(mean_lat, 2)
        },
        "component_breakdown_avg_ms": {
            "avg_retrieval_ms": round(float(np.mean(retrieval_latencies)), 2),
            "avg_generation_ms": round(float(np.mean(generation_latencies)), 2)
        },
        "queries": detailed_results
    }

    with open(BENCHMARK_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Benchmark finished. P50: {p50:.2f}ms | P70: {p70:.2f}ms | P100: {p100:.2f}ms")
    logger.info(f"Results saved to {BENCHMARK_OUTPUT_FILE}")
    return summary

if __name__ == "__main__":
    run_benchmark(num_queries=30)
