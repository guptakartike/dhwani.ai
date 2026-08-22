# Architecture — Voice-Enabled RAG

## Pipeline

```
Audio input (mic, streamed)
        │
        ▼
Sarvam STT (saaras:v3-realtime, stream_type=fast)
        │  → final transcript (Hindi text)
        ▼
Query embedding (bge-small, local)
        │  → query vector
        ▼
Qdrant retrieval (3 chunking strategies, ranked)
        │  → top-k chunks + confidence scores
        ▼
   ┌────┴────┐
   │  Guardrail gate │  ← confidence below threshold?
   └────┬────┘
        │ pass                          │ fail
        ▼                                ▼
Groq generation                  Refusal response
(Llama 3.1, grounded prompt)     ("not found in context")
        │
        ▼
   Guardrail check on output
   (off-topic / unsafe / ungrounded?)
        │
        ▼
   Final answer (JSON: answer, confidence, sources)
```

The guardrail is a **hard gate at two points**: before generation (retrieval
confidence check) and after generation (output validation). Both must be able to
independently veto and return a refusal — this is graded, not cosmetic.

## Component contracts

### 1. STT service (`stt.py`)
- Input: raw audio stream (WebSocket)
- Output: `TranscriptResult(text: str, is_final: bool, confidence: float)`
- Must handle partial + final transcripts; only final transcripts trigger retrieval
- Retry once on connection failure, then surface a clean error upstream

### 2. Embedding service (`embed.py`)
- Input: `str` (query text)
- Output: `list[float]` (vector, dimension matches bge-small)
- No network round-trip — must run locally for latency budget

### 3. Retrieval service (`retrieval.py`)
- Input: query vector + optional metadata filters
- Output: `list[RetrievedChunk(text: str, score: float, strategy: str, source_id: str)]`
- Must query across at least 3 chunking-strategy collections and return the best
  results, tagged with which strategy produced them (for benchmarking/reporting)

### 4. Generation service (`generate.py`)
- Input: query text + retrieved chunks
- Output: `GenerationResult(answer: str, grounded: bool, sources: list[str])`
- Prompt must instruct the model to answer only from provided context and to say
  explicitly when it cannot

### 5. Guardrail service (`guardrails.py`)
- Pre-generation gate: reject if top retrieval score < threshold (tune empirically)
- Post-generation gate: reject if output looks ungrounded, off-topic, or unsafe
- Both gates return a structured `RefusalReason` when triggered — log every refusal,
  don't silently drop it

### 6. Orchestration / harness (`main.py`, FastAPI app)
- Wires all five services together behind one endpoint
- Wraps external calls (STT, Groq) in `tenacity` retries
- All request/response bodies are Pydantic models — no raw dicts crossing boundaries
- Structured error recovery: empty retrieval → skip generation, return refusal;
  STT failure → retry once, then clean error; Groq failure → retry once, then
  fallback refusal response

### 7. Benchmarking (`benchmark.py`)
- Runs a fixed batch of 30–50 test queries through the post-transcription pipeline
  (retrieval + generation only, per the task's latency scope)
- Logs per-query latency, computes and prints P50/P70/P100
- Output should be saved to a file (`benchmark_results.json` or similar) that gets
  included in the submission

## Suggested repo structure

```
/app
  main.py              # FastAPI app, orchestration
  stt.py                # Sarvam integration
  embed.py              # bge-small embedding
  retrieval.py          # Qdrant client + chunking strategies
  generate.py            # Groq integration
  guardrails.py          # confidence gate + output validation
  schemas.py              # Pydantic models for all boundaries
  benchmark.py             # latency test harness
/data
  ingest.py                # loads MSMARCO-XI Hindi subset, builds chunk collections
/tests
  test_guardrails.py         # off-topic, unsafe, ungrounded query cases
```

## Dataset

`ai4bharat/MSMARCO-XI`, Hindi subset. Each example has a query, one or more answers,
and a set of passages. Passages are pre-segmented — chunking strategies should build
on top of these, not re-chunk raw unstructured text from scratch.
