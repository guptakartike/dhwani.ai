# Voice-Enabled RAG Pipeline (HH Goa 2026 Task 2)

High-performance, voice-enabled Retrieval-Augmented Generation (RAG) system for Hindi, built with Sarvam AI STT, local `bge-small` embeddings, multi-strategy Qdrant retrieval, Groq (Llama 3.1 8B) generation, and a strict two-tier hard guardrail gate.

## 🚀 Key Features

- **Sarvam AI Streaming STT**: Real-time Hindi speech-to-text (`saaras:v3-realtime`) with `stream_type="fast"`.
- **Multi-Strategy Retrieval**: 3 chunking strategies indexed in separate Qdrant collections:
  1. Native Passage-level chunking
  2. Sentence-Window ($\pm 1$ sliding window context)
  3. Semantic Sentence chunking (cosine similarity grouping)
- **Local Embedding Overhead Prevention**: `bge-small` model running locally to meet the sub-200ms retrieval+generation budget.
- **Two-Tier Hard Guardrails**:
  - **Pre-Generation Gate**: Refuses queries with low retrieval confidence scores ($< 0.45$) or harmful content.
  - **Post-Generation Gate**: Refuses ungrounded or out-of-context LLM outputs.
- **Latency Benchmarking**: Automated test suite measuring P50, P70, and P100 post-transcription latency.

---

## 🛠️ Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API credentials in `.env`:
   ```env
   SARVAM_API_KEY=your_sarvam_api_key
   GROQ_API_KEY=your_groq_api_key
   QDRANT_URL=http://localhost:6333
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Application

### 1. Start FastAPI Web Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints:
- **`GET /`**: API Status & Configuration
- **`POST /api/query`**: Execute RAG Pipeline for text queries
- **`WS /ws/rag`**: Real-time Audio Stream WebSocket Endpoint

---

## 🧪 Testing & Benchmarking

### Run Guardrail Unit Tests
```bash
pytest tests/test_guardrails.py
```

### Run Latency Benchmark (P50, P70, P100)
```bash
python app/benchmark.py
```
This will run 30 queries against the MSMARCO-XI Hindi dataset, log per-query latency breakdowns, compute percentiles, and output `benchmark_results.json`.
