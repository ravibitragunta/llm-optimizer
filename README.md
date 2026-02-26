# llm-optimizer

> **Proving that a Context Graph + Exponential Decay Router reduces LLM compute costs in multi-turn code generation sessions — without sacrificing code quality.**

---

## The Problem

Every time you send a message in a multi-turn LLM coding session, the model receives the **entire conversation history** again from the beginning. By turn 9 of a typical Spring Boot session, 99% of the prompt is old repetitive history — and the model still pays the full attention cost over all of it.

This is **context explosion**: prompt tokens grow linearly, compute cost grows quadratically (O(n²) in transformers), and generation speed degrades.

```
Turn 1:   78 tokens   sent to model
Turn 5:  5,281 tokens sent to model  (67× more)
Turn 9: 11,068 tokens sent to model  (141× more)
```

Eventually the context window overflows, the session breaks, and the user starts over — losing all accumulated session knowledge.

---

## The Idea

Instead of dumping raw conversation history, maintain a **living graph** of what the session actually contains:

- **Active concepts** (entities referenced recently, weighted by recency)
- **Established decisions** ("we are using NamedParameterJdbcTemplate", "constructor injection throughout")
- **Detected gaps** (things in the domain that haven't been addressed yet)
- **Contradictions** (conflicting decisions the model should not make)

Then compress this into a **bounded, structured context block** — always under a fixed token budget — and send that to the model instead.

The graph **decays** over time (concepts fade if not referenced) so old, settled decisions take up less space while current active work stays hot.

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────┐
│  ContextGraph (brain/graph.py)                   │
│  - Extract entities via regex + spaCy            │
│  - Update node darkness (recency weight)         │
│  - Apply exponential decay each turn             │
│  - Detect decisions, contradictions              │
└──────────────┬───────────────────────────────────┘
               │ active_nodes
               ▼
┌──────────────────────────────────────────────────┐
│  DecayRouter (brain/router/decay_router.py)      │
│  - Encode graph as 256-dim state vector          │
│  - Update state: s = s × decay + signal × rate  │
│  - Score concepts by dot product with state      │
│  - Detect current domain (spring_jdbc, etc.)     │
└──────────────┬───────────────────────────────────┘
               │ router state
               ▼
┌──────────────────────────────────────────────────┐
│  Retriever (brain/router/retriever.py)           │
│  - Jaccard similarity against stored patterns   │
│  - Load domain knowledge from Redis seeds       │
│  - Surface gaps and contradictions              │
└──────────────┬───────────────────────────────────┘
               │ retrieved context
               ▼
┌──────────────────────────────────────────────────┐
│  PromptBuilder (brain/prompt.py)                 │
│  - Assemble: decisions → request → active ctx   │
│  - Hard-cap at MAX_PROMPT_TOKENS                │
│  - Priority: decisions > request > graph > hist │
└──────────────┬───────────────────────────────────┘
               │ bounded prompt
               ▼
        Qwen2.5-Coder-3B (local, llama-cpp-python)
               │ response
               ▼
┌──────────────────────────────────────────────────┐
│  Memory (brain/memory.py)                        │
│  - Save graph + router + history to Redis       │
│  - Pipeline atomic writes                       │
│  - Pattern store if confidence threshold met    │
└──────────────────────────────────────────────────┘
```

---

## Benchmark Results (v0.1 — 10-turn DAO Suite, CPU)

**Model:** Qwen2.5-Coder-3B-Instruct-Q4_K_M · **Hardware:** CPU only · **Turns:** 10 (Spring Boot DAO → Service → Controller stack)

| Turn | Baseline Tokens | Brain Tokens | Reduction |
|-----:|----------------:|-------------:|----------:|
| 1    | 78              | 282          | —         |
| 3    | 2,769           | 1,650        | 1.7×      |
| 5    | 5,281           | 2,800        | 1.9×      |
| 7    | 8,231           | 2,801        | 2.9×      |
| 9    | 11,068          | 2,800        | **4.0×**  |

| Metric | Baseline | Brain |
|--------|----------|-------|
| Avg prompt tokens | 5,500 | **2,313** |
| Peak prompt tokens | 11,068 | **2,801** |
| Context growth | Linear explosion | **Flat (capped)** |
| Estimated compute reduction | — | **~6×** (token² ratio) |
| Brain avg quality | — | 5.4 / 11 |

> The baseline **crashed at turn 4** on the default 4096-token context window. The brain kept running to turn 10 within budget.

---

## Project Structure

```
llm-optimizer/
├── brain/
│   ├── config.py           # All tunable parameters via .env
│   ├── graph.py            # ContextGraph: entities, decay, serialization
│   ├── prompt.py           # Token-budgeted prompt assembler
│   ├── memory.py           # Redis session persistence
│   ├── brain.py            # Orchestrator — the think() pipeline
│   └── router/
│       ├── decay_router.py # 256-dim exponential decay state router
│       └── retriever.py    # Jaccard RAG + gap/contradiction detection
├── seed/
│   ├── spring_jdbc.json    # Domain knowledge: Spring JDBC
│   ├── spring_boot.json    # Domain knowledge: Spring Boot
│   └── spring_security.json
├── benchmarks/
│   ├── runner.py           # Head-to-head benchmark (baseline vs brain)
│   └── conversations/
│       └── dao_suite.json  # 10-turn DAO benchmark conversation
├── tests/
│   └── quality/
│       ├── scorer.py       # Code quality rubric (0–11 score)
│       └── coherence.py    # Context coherence checker
├── calibration/
│   └── calibrate.py        # Parameter sweep runner
├── data/
│   ├── models/             # Local GGUF model files
│   └── results/            # Benchmark CSVs (baseline/ and brain/)
├── seed.py                 # One-shot Redis domain seeder
├── requirements.txt
├── .env.example
└── Approach.md             # Full technical design document
```

---

## Running It

**Prerequisites:**
- Python 3.12+
- Redis (compiled from source or via apt)
- ~2.5 GB disk for the GGUF model

```bash
# 1. Install dependencies
pip install --user --break-system-packages -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure
cp .env.example .env
# Edit .env: set MODEL_PATH to your downloaded .gguf file

# 3. Start Redis and seed domain knowledge
data/redis-stable/src/redis-server --daemonize yes
python seed.py

# 4. Run benchmark
PYTHONPATH=$(pwd) python benchmarks/runner.py
```

---

## Tunable Parameters (`.env`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MODEL_PATH` | `data/models/...gguf` | Path to local GGUF model |
| `MODEL_N_CTX` | `4096` | LLM context window |
| `MODEL_N_THREADS` | `4` | CPU threads (set to physical core count) |
| `MAX_PROMPT_TOKENS` | `700` | Brain prompt budget cap |
| `DARKNESS_DECAY` | `0.90` | How fast graph nodes fade per turn |
| `DARKNESS_INCREMENT` | `0.3` | How much a mentioned node brightens |
| `DARKNESS_THRESHOLD` | `0.05` | Below this, node is pruned from graph |
| `ROUTER_DECAY_RATE` | `0.85` | State vector geometric decay |
| `ROUTER_UPDATE_RATE` | `0.15` | New signal weight in state |
| `ROUTER_STATE_DIM` | `256` | Dimensionality of router state |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum Jaccard score for pattern match |

---

## Known Issues / Current Limitations

- **Java/Spring-specific entity extraction** — the regex and keyword patterns are domain-specific. Generalising to other languages requires pluggable extractors.
- **Quality scorer calibration** — the 0–11 rubric was calibrated for the DAO suite; late turns (exception handling, `@ControllerAdvice`) are under-scored due to sparse keyword coverage.
- **Coherence metric** — uses exact string matching; lemmatisation would make it more accurate.
- **CPU-only** — all benchmarks run on CPU. Brain overhead (graph + retrieval) is ~10–20ms/turn, negligible on GPU but visible on CPU.

---

## Next Steps

### Immediate — Quality Improvement
- [ ] Raise `MAX_PROMPT_TOKENS` from 2800 → 3500 in default config
- [ ] Fix quality scorer keyword sets for turns 7–10 (exception handling, validation)
- [ ] Improve system prompt: add incremental build instructions ("do not rewrite existing classes")
- [ ] Add `SPRING_EXCEPTION_PATTERNS` entity list for `@ControllerAdvice`, `ResponseEntity`, `HttpStatus`
- [ ] Replace coherence exact-match with spaCy lemmatised substring matching

### Near-term — CPU Throughput
- [ ] Tune `n_threads` docs to physical core count (hyperthreading hurts llama.cpp)
- [ ] Enable `use_mlock` and tune `n_batch` in llama-cpp-python config
- [ ] Benchmark Q4_K_S vs Q4_K_M quantisation (faster, marginal quality tradeoff)
- [ ] Dynamic `max_tokens` per turn type (short confirmations don't need 512 token budget)
- [ ] Cache system prompt KV state at startup (pay prompt processing cost once, not per turn)

### Medium-term — Language Generalisation
- [ ] Refactor entity extraction into a `BaseExtractor` plugin interface
- [ ] Implement `PythonFastAPIExtractor`, `TypeScriptNestJSExtractor`
- [ ] Author seed JSONs for Python/FastAPI, TypeScript/NestJS, Ruby/Rails domains
- [ ] Refactor `SpringBootQualityScorer` into a `BaseQualityScorer` plugin
- [ ] Make the router domain map config-driven (not hardcoded in `decay_router.py`)
- [ ] Benchmark suite for Python/FastAPI (equivalent of `dao_suite.json`)

### Longer-term — Architecture
- [ ] System prompt KV-cache pre-computation at Brain init
- [ ] Async graph ingestion (overlap with LLM generation for concurrency)
- [ ] Multi-session pattern accumulation — patterns learned in one project inform another
- [ ] Export graph as structured "session memory" file (portable, inspectable)
- [ ] Web UI for inspecting live graph state, darkness values, active decisions during a session

---

## Design Philosophy

- **No embeddings, no PyTorch, no vector databases.** The retrieval uses Jaccard set similarity — fast, deterministic, no GPU required.
- **Local-first.** The entire stack runs offline with a GGUF model. No API keys, no rate limits, no usage costs.
- **Mathematically grounded decay.** Concept relevance follows a geometric series, not heuristic rules.
- **Pluggable by design** (target state) — entity extraction, quality scoring, and domain seeds are meant to be authored per language, not hardcoded.
- **Measure everything.** Every `think()` call emits a full metrics dict: tokens, tps, graph size, coherence, gaps found, retrieval latency.

---

## References

- [Approach.md](./Approach.md) — Full technical design document with all mathematical formulations
- `data/results/` — Raw benchmark CSVs from all runs
