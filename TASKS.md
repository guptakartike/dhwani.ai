# Tasks — work through in order, phase by phase

Report status after each phase before moving to the next.

## Phase 0 — Setup
- [ ] Initialize repo with structure from ARCHITECTURE.md
- [ ] Install dependencies from TECH_STACK.md
- [ ] Load `.env` with SARVAM_API_KEY, GROQ_API_KEY, QDRANT_URL
- [ ] Download `ai4bharat/MSMARCO-XI` Hindi subset via `datasets` library, inspect schema
- **Acceptance:** dataset loads, prints a sample query + passages + answer correctly

## Phase 1 — Data ingestion & chunking
- [ ] Implement passage-level chunking (baseline — use provided passages as-is)
- [ ] Implement sentence-window chunking (split passages into sentences, ±1 window)
- [ ] Implement semantic chunking (embed sentences, merge/split on similarity breaks)
- [ ] Tag all chunks with metadata: `source_lang`, `query_id`, `strategy`, position
- [ ] Embed all chunks with bge-small, load into separate Qdrant collections per strategy
- **Acceptance:** 3 populated Qdrant collections, each queryable, each chunk carries
  strategy + metadata tags

## Phase 2 — Retrieval
- [ ] Implement `retrieval.py`: given a query vector, search across all 3 collections
- [ ] Return top-k results merged/ranked with strategy attribution
- [ ] Add basic hybrid filtering (metadata-aware) as a bonus if time allows
- **Acceptance:** given a sample query, retrieval returns relevant passages with
  correct source attribution and scores

## Phase 3 — STT integration
- [ ] Wire Sarvam `saaras:v3-realtime` streaming with `stream_type=fast`
- [ ] Handle partial vs. final transcripts correctly
- [ ] Add retry-once-then-fail behavior on connection drop
- **Acceptance:** speaking a Hindi query produces a correct final transcript that
  triggers the retrieval step

## Phase 4 — Generation
- [ ] Wire Groq client, Llama 3.1 8B
- [ ] Write a strictly grounded prompt (answer only from provided context, say when
  it cannot)
- [ ] Structure output as validated Pydantic model (answer, confidence, sources)
- **Acceptance:** given retrieved chunks + query, produces a grounded answer citing
  source chunks; produces an explicit "cannot answer" when context is irrelevant

## Phase 5 — Guardrails
- [ ] Pre-generation gate: reject if top retrieval score below threshold (tune this
  empirically against real queries, don't guess a number)
- [ ] Post-generation gate: validate output isn't off-topic/unsafe/ungrounded
- [ ] Write test cases: 1 off-topic query, 1 unsafe/inappropriate input, 1 query with
  no good context match — confirm all three are refused, not answered
- **Acceptance:** all 3 test cases correctly produce a refusal, not a hallucinated
  answer

## Phase 6 — Harness
- [ ] Wrap Sarvam and Groq calls in `tenacity` retries
- [ ] Ensure every service boundary uses Pydantic models, not raw dicts
- [ ] Add structured error recovery: empty retrieval → skip generation, return
  refusal; STT/Groq failure → retry once then clean error response
- **Acceptance:** killing network mid-call to Sarvam or Groq results in a retry then
  a clean structured error, not a crash

## Phase 7 — Benchmarking
- [ ] Build `benchmark.py`: runs 30–50 real test queries through retrieval +
  generation (post-transcription scope, per task spec)
- [ ] Log per-query latency, compute P50/P70/P100
- [ ] Save results to `benchmark_results.json`
- **Acceptance:** benchmark completes on a real batch, numbers are saved and
  reproducible, not a single best-case run

## Phase 8 — Deployment
- [ ] Deploy to Railway/Render/Fly.io free tier
- [ ] Confirm live URL works end-to-end from a fresh browser/session
- [ ] Push final code to GitHub
- **Acceptance:** live link works for someone who isn't you, testing cold

## Phase 9 — Submission prep
- [ ] Record 90s process video
- [ ] Record end-to-end demo video (must show a guardrail refusal in action)
- [ ] Confirm benchmark numbers are included in submission materials
- **Acceptance:** everything in PLANNING.md's submission checklist is checked off
