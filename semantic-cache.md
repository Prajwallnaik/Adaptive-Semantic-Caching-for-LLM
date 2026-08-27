# Semantic Embedding Cache — Project Brain

## 1. What this project is

A caching layer that sits in front of an LLM API to cut cost and latency by detecting
*semantically similar* queries — not just exact repeats — and serving a stored answer
instead of calling the LLM again. Wrapped with production-grade monitoring so the
project demonstrates observability and correctness testing, not just "it caches."

**Core value proposition (the one-liner):**
> Reduce LLM API cost and latency by serving cached answers for semantically similar
> queries, with hybrid retrieval, metadata scoping, and full observability — measured,
> not just claimed.

---

## 2. Core architecture (request flow)

```
Client query (normalized, tenant-scoped)
        │
        ▼
Metadata-scoped hybrid search
  (BM25 + vector, RRF fused, filtered by user_id/model/context_version FIRST)
        │
   ┌────┴────┐
   ▼         ▼
Above       Below
threshold   threshold
   │         │
   ▼         ▼
Verify      Call LLM provider (Nemotron via NVIDIA NIM)
borderline       │
hits (judge)     ▼
   │        Async write-behind to cache
   ▼        (embedding + BM25 terms + metadata + TTL, non-blocking)
Return
cached
answer

Both paths → Observability layer (Prometheus + Grafana)
```

**Key design decision:** metadata filtering happens *before/during* retrieval, not
after ranking. Filtering after ranking wastes top-k slots on candidates that get
discarded — pre-filtering shrinks the search space and is both cheaper and safer.

---

## 3. Why each major decision was made

- **Hybrid search (dense + sparse) over pure embedding similarity**
  Pure cosine similarity conflates topically-similar-but-factually-different queries
  (e.g. "Q1 2024 revenue" vs "Q1 2023 revenue" embed close together but need
  different answers). BM25/sparse catches numbers, entities, dates that embeddings blur.
  Fusion via RRF or weighted `α·dense + (1-α)·sparse`.

- **Metadata filtering**
  Prevents cross-tenant cache leakage (`user_id`), prevents serving one model's
  cached answer for a different model's request (`model`), and enables cache
  invalidation when the system prompt/context changes (`context_version`).

- **Verification step on borderline hits**
  Don't trust the threshold blindly. Scores within a small tolerance band of the
  threshold get a cheap secondary check (small/fast LLM judge call or rule-based
  overlap check) before being served. This is the difference between "cache based
  on vibes" and "cache with a correctness guarantee" — the strongest interview
  talking point in the whole project.

- **Async write-behind caching**
  Client gets the LLM's answer immediately; the cache write happens in a background
  task afterward. Real, demonstrable latency win.

- **Three-part eviction: TTL + LRU + versioning**
  Each solves a different failure mode — none alone is sufficient:
  - TTL → time decay of correctness (stale prices, news)
  - LRU → storage budget (cap collection size)
  - context_version → structural change (prompt/context changed, old entries
    should never match again, regardless of age or usage)
  TTL + LRU run as scheduled sweeps; versioning is a free filter at query time.

- **Eval harness over just "does it run"**
  A labeled set of paraphrase pairs (should hit) and near-miss pairs (should NOT
  hit) run through the real pipeline at multiple thresholds, scored on
  precision/recall. This is what proves the cache is *safe*, not just functional.

---

## 4. Tech stack

| Layer | Choice |
|---|---|
| API layer | FastAPI |
| LLM provider | `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`, OpenAI-compatible, direct — NOT OpenRouter) |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, local/free) |
| Vector + hybrid search | Qdrant (self-hosted, Docker) — native hybrid (dense + sparse) + RRF + metadata payload filtering |
| Cache metadata store | Qdrant payload fields (user_id, model, context_version, expires_at, last_accessed_at) |
| Async task queue | FastAPI `BackgroundTasks` (MVP) → Celery + Redis (stretch) |
| Verification / judge | Same Nemotron model, short cheap prompt, or rule-based overlap check |
| Metrics instrumentation | `prometheus-fastapi-instrumentator` + `prometheus_client` |
| Metrics collection | Prometheus |
| Dashboards | Grafana, auto-refresh 5-10s |
| Containerization | Docker Compose: `app` (own FastAPI code) + `qdrant` + `prometheus` + `grafana` (all off-the-shelf images) |
| Frontend / chatbot | Streamlit or React — just another client hitting `/query`, no separate metrics pipeline needed |
| Load testing | Custom `asyncio` script (or Locust) |
| Eval harness | Python + pandas over labeled JSON pairs |

NVIDIA client shape:
```python
from openai import OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="YOUR_NVIDIA_API_KEY"
)
response = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    messages=[{"role": "user", "content": query}]
)
```

**Free-tier framing note:** NVIDIA's free tier has rate limits — frame the cache as
not just a cost optimizer but a way to stay under quota. Track
`requests_saved_from_quota` as a metric.

---

## 5. Live dashboard behavior (chatbot → Grafana)

No separate pipeline needed. The chatbot (Streamlit/React) only talks to `/query`.
Every request — whether from the load test script or a human typing — increments
the same Prometheus counters. Flow:

```
Chatbot UI → FastAPI /query → cache hit/miss → Prometheus counters updated
                                                       ↓
                                    Prometheus scrapes /metrics (every 5s)
                                                       ↓
                                    Grafana queries Prometheus, auto-refreshes
```

Nice-to-have: `/stats` endpoint (separate from `/metrics`, meant for UI polling)
so the chatbot sidebar can show a live "session hit rate" indicator, and a
per-message badge like "cache hit ⚡ 12ms" / "cache miss — called LLM 1.8s".

---

## 6. Testing strategy (three distinct questions, three distinct folders)

| Folder | Question it answers |
|---|---|
| `tests/unit/` | Does each function behave correctly in isolation (mocked deps)? |
| `tests/integration/` | Does the real pipeline (real Qdrant, mocked LLM) behave correctly end-to-end? |
| `eval/` | Does the cache make semantically *correct* decisions (precision/recall)? |
| `loadtest/` | Does it perform well under realistic traffic (the portfolio numbers)? |

**Unit tests cover:** threshold hit/miss/verify logic, RRF fusion math, metadata
filter construction, TTL/LRU eviction selection logic.

**Integration tests cover:** write-then-paraphrase-hits, unrelated-query-misses,
expired-TTL-not-served, multi-tenant isolation (user A's cache never leaks to user B).
Use `testcontainers-python` to spin up real Qdrant per test run.

**Eval harness:** `paraphrase_pairs.json` (should hit) + `near_miss_pairs.json`
(should NOT hit) → run through real pipeline at multiple thresholds → precision/recall
curve → justifies the chosen threshold.

**Load test (the headline-number generator):**
- Realistic traffic mix: ~25% exact repeats, ~35% paraphrases, ~40% novel queries
- 2,000+ requests via `asyncio`/httpx against the live `/query` endpoint
- Compute: hit-rate convergence over time, latency percentiles (p50/p95/p99) split
  by hit/miss, total cost savings (misses × cost-per-call vs. no-cache baseline)
- Target headline sentence for README:
  > "Under a realistic traffic mix, cache hit rate converged to ~65% within the
  > first 500 requests, cutting p95 latency from 2.1s to 40ms and reducing LLM
  > API calls by ~60%."

---

## 7. File structure

```
semantic-cache/
├── app/
│   ├── main.py                       # /query, /metrics
│   ├── config.py                     # env vars, thresholds
│   ├── api/                          # routes.py, schemas.py
│   ├── core/                         # embeddings, hybrid_search, metadata_filter,
│   │                                    threshold, verification, llm_client
│   ├── cache/                        # store, write_behind, eviction
│   ├── monitoring/                   # metrics.py (Prometheus)
│   └── jobs/                         # scheduler.py (periodic eviction)
├── chatbot/                          # streamlit_app.py or react-app/
├── eval/                             # datasets/, run_eval.py, results/
├── loadtest/                         # generate_queries.py, run_load_test.py,
│                                        analyze_results.py
├── tests/
│   ├── unit/
│   └── integration/                  # + conftest.py (testcontainers Qdrant fixture)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/, provisioning/
├── docker/Dockerfile
├── docker-compose.yml                # app + qdrant + prometheus + grafana
├── requirements.txt
├── .env.example
└── README.md
```

**Design rationale:**
- `app/core/` (retrieval + decision logic) is split from `app/cache/` (persistence +
  lifecycle) so hit/miss logic is testable independent of storage.
- `eval/` and `loadtest/` are top-level siblings of `app/`, not nested inside —
  they're external consumers of the API, not part of the service itself.
- `monitoring/grafana/dashboards/cache_dashboard.json` should be exported once built
  so `docker-compose up` gives a fully configured dashboard, not an empty instance.

---

## 8. Build order (suggested)

1. `app/core/embeddings.py` + `app/cache/store.py` — basic embed + Qdrant write/read
2. `app/core/hybrid_search.py` + `app/core/metadata_filter.py` — hybrid retrieval, scoped
3. `app/core/threshold.py` — hit/miss/verify decision
4. `app/core/llm_client.py` — Nemotron via NVIDIA NIM
5. `app/cache/write_behind.py` — async cache write on miss
6. `app/cache/eviction.py` + `app/jobs/scheduler.py` — TTL + LRU + versioning sweep
7. `app/monitoring/metrics.py` — Prometheus instrumentation
8. `docker-compose.yml` — wire app + qdrant + prometheus + grafana together
9. `tests/unit/` as each component above is written
10. `tests/integration/` once Qdrant is wired
11. `eval/` once hybrid search + threshold are working
12. `chatbot/` — Streamlit/React hitting the live `/query`
13. `loadtest/` — once the whole pipeline is live, run for the portfolio numbers
14. Grafana dashboard build + export to `monitoring/grafana/dashboards/`
15. README with architecture diagram + headline numbers from step 13

---

## 9. Open decisions to revisit while building

- Exact similarity threshold value and verification tolerance band (determine via
  the eval harness sweep, not guessed upfront)
- Whether to add Redis as an L1 exact-match cache in front of Qdrant (stretch goal)
- Celery + Redis upgrade for async writes if `BackgroundTasks` proves insufficient
  under load
- Whether verification calls should use Nemotron itself or a cheaper/local model,
  given free-tier rate limits
