# Tech stack

Do not substitute any of these without flagging it first — Sarvam and Groq in
particular are load-bearing for both the task's technical requirements and the
latency target.

| Layer | Tool | Notes |
|---|---|---|
| Speech-to-text | **Sarvam AI**, model `saaras:v3-realtime` | Required option per task brief (Sarvam or ElevenLabs — Sarvam chosen for Indic-language fit with the dataset). Set `stream_type="fast"` explicitly — the default buffers 1000ms and is too slow. |
| Language | **Hindi (`hi`)** | Matches dataset subset and team fluency |
| Embeddings | **bge-small** (local, via `sentence-transformers` or `FlagEmbedding`) | Free, fast, avoids an API round-trip on the query-latency-critical path |
| Vector DB | **Qdrant** (self-hosted via Docker, or Qdrant Cloud free tier) | Sub-10ms search, supports payload filtering for metadata-aware retrieval |
| Generation | **Groq API**, Llama 3.1 8B (fall back to 70B only if latency budget allows) | Required for hitting the sub-200ms retrieval+generation target — standard LLM APIs are too slow for this |
| Backend | **FastAPI** (Python 3.11+) | Async-native, works well with the above |
| Retries | **tenacity** | Wraps Sarvam and Groq calls |
| Schemas | **Pydantic v2** | Every request/response boundary must be a validated model |
| Deployment | **Railway, Render, or Fly.io** (free tier) | Needs to produce a real live link per team member |
| Repo hosting | **GitHub** | One repo per team member (can share a common upstream, but each person needs their own working fork/deployment) |

## Environment variables needed

```
SARVAM_API_KEY=
GROQ_API_KEY=
QDRANT_URL=            # if using Qdrant Cloud, otherwise localhost
QDRANT_API_KEY=        # if using Qdrant Cloud
```

## Python dependencies (starting point)

```
fastapi
uvicorn
pydantic>=2
tenacity
qdrant-client
sentence-transformers
groq
websockets
python-dotenv
datasets          # for loading ai4bharat/MSMARCO-XI from HuggingFace
```
