# Planning — Voice RAG (HH Goa 2026, Task 2)

**Deadline: Aug 22, 2026, 11:59 PM IST. No resubmissions.**

## Context

Team of 3, each member must independently submit a working instance (all-or-nothing
selection rule — every member must personally clear the task, not just contribute to
one shared submission). This doc set is being fed to Antigravity by each teammate
separately; each instance should build the full pipeline, but real-world division of
deep ownership is:

- **Person A** — STT integration, audio handling
- **Person B** — Chunking strategies, Qdrant retrieval
- **Person C** — Generation, guardrails, harness, benchmarking

Even if Antigravity builds the whole thing per-instance, each teammate should be able
to explain and defend their owned slice in depth — that's part of what's being judged.

## Milestones

| Day | Goal |
|---|---|
| Aug 19 (tonight) | API keys obtained (Sarvam, Groq), repo skeleton up, dataset loaded |
| Aug 20 | Core pipeline working end-to-end (may be rough/unoptimized) |
| Aug 21 | Harness (retries, structured I/O, error recovery) added; benchmarking run producing real P50/P70/P100; each person deploys their own live instance |
| Aug 22 AM | Guardrail testing — off-topic, unsafe, ungrounded queries confirmed to actually refuse. Build frozen by early afternoon. |
| Aug 22 PM | Videos recorded and posted, submission form filled |

## Definition of done (per person)

- [ ] Pipeline runs end-to-end locally
- [ ] Deployed to a live public URL
- [ ] GitHub repo is clean and pushed
- [ ] Benchmark script run against 30–50 real queries, results saved
- [ ] At least 3 chunking strategies implemented and comparable
- [ ] Guardrail demonstrably refuses at least one real off-topic/ungrounded query
- [ ] 90-second process video recorded
- [ ] End-to-end demo video recorded

## Submission checklist (mandatory, per person)

- [ ] GitHub repo link
- [ ] Live working link
- [ ] Video 1 (90s process video) posted to Instagram **and** X
- [ ] Video 2 (demo video) posted to Instagram **and** X
- [ ] `#RAGInGoa` on every single post, every platform
- [ ] At least one Instagram account public
- [ ] Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58

## Known ambiguities (worth confirming with organizers if time allows)

- Whether teammates can share a codebase with individual deployments, or must build
  fully separate projects — lean toward "share core, deploy separately, own a
  distinct piece" as the safer interpretation.
