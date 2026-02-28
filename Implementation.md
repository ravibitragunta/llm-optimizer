# Implementation Guide
## Brain MVP — Context Graph + Exponential Decay Router

> This document covers every file in the system, the math behind the key components, and both optimisation passes that were applied after the initial benchmark.

---

## Table of Contents

1. [Overall Architecture](#overall-architecture)
2. [Data Flow — One Turn](#data-flow--one-turn)
3. [File-by-File Reference](#file-by-file-reference)
   - [brain/config.py](#brainconfigpy)
   - [brain/graph.py](#braingraphpy)
   - [brain/router/decay_router.py](#brainrouterdecay_routerpy)
   - [brain/router/retriever.py](#brainrouterretrieverpy)
   - [brain/prompt.py](#brainpromptpy)
   - [brain/memory.py](#brainmemorypy)
   - [brain/brain.py](#brainbrainpy)
   - [seed.py & seed/*.json](#seedpy--seedjson)
   - [tests/quality/scorer.py](#testsqualityscorerpy)
   - [tests/quality/coherence.py](#testsqualitycoherencepy)
   - [benchmarks/runner.py](#benchmarksrunnerpy)
4. [Optimisation Pass 1 — Quality](#optimisation-pass-1--quality)
5. [Optimisation Pass 2 — CPU Throughput](#optimisation-pass-2--cpu-throughput)
6. [Configuration Reference](#configuration-reference)
7. [Redis Key Schema](#redis-key-schema)

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          brain/brain.py  (Brain.think)                  │
│  Cloud-only mode: self.llm = None  |  All inference → Cloudflare Workers│
│                                                                         │
│  User Input                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────┐    active_nodes    ┌─────────────────────────┐    │
│  │  graph.py        │ ─────────────────▶ │  decay_router.py        │    │
│  │  ContextGraph    │                    │  DecayRouter             │    │
│  │  - extract       │ ◀───────────────── │  - 256-dim state vector │    │
│  │  - decay         │  get_active_       │  - geometric decay      │    │
│  │  - summarize     │  concepts()        │  - domain detection     │    │
│  └─────────────────┘                    └────────────┬────────────┘    │
│      │                                               │ router.state    │
│      │ graph_summary                                 ▼                  │
│      │                                  ┌─────────────────────────┐    │
│      │                                  │  retriever.py           │    │
│      │                                  │  - Jaccard similarity   │    │
│      │                                  │  - Redis pattern store  │    │
│      │                                  │  - gap/contradiction     │    │
│      │                                  └────────────┬────────────┘    │
│      │                                               │ retrieved_data  │
│      ▼                                               ▼                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  prompt.py  —  PromptBuilder.build()                             │  │
│  │                                                                  │  │
│  │  ── PROTECTED (never truncated) ──────────────────────────────   │  │
│  │  Session Header → Universal Adaptive Block → Decisions           │  │
│  │  → Compressed History (~15 tok/turn) → Current Request          │  │
│  │                                                                  │  │
│  │  ── OPTIONAL (trimmed if budget exhausted) ────────────────────  │  │
│  │  Active Graph Context → Gaps / Contradictions                   │  │
│  │                         ┌───────────────────────────────┐        │  │
│  │                         │ Budget: get_token_budget(domain)│       │  │
│  │                         │ Default 2800, java/python 3500 │        │  │
│  │                         └───────────────────────────────┘        │  │
│  └─────────────────────────┬────────────────────────────────────────┘  │
│                             │ bounded prompt (ChatML format)            │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  clients/cloud_client.py  —  CloudClient                         │  │
│  │  _cloudflare_coding()  →  POST https://qwen-coder-worker...      │  │
│  │  Request: {"model": "qwen-2.5-coder-32b", "prompt": "<ChatML>"} │  │
│  │  Response key: "text"                                            │  │
│  │  summarize_history()  every 5 turns  →  same worker              │  │
│  └─────────────────────────┬────────────────────────────────────────┘  │
│                             │ response_text                             │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────┐                      │
│  │  memory.py  —  Memory.save_session()         │                      │
│  │  Redis pipeline: graph + router + history    │                      │
│  └──────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — One Turn

```
think(user_input)
│
├── 0a. plugin_registry.detect(user_input) → plugin
│         Identifies domain plugin: java_spring / sap / general
│
├── 0b. Every 5 turns: cloud_client.summarize_history()
│         POST to Cloudflare Worker with JSON extraction prompt
│         Injects decisions/facts/contradictions into graph
│         ⚠ Currently returns truncated JSON — caught gracefully
│
├── 1. graph.ingest(user_input, turn_N)
│         Extract entities via regex + spaCy + keyword lists
│         Find/create nodes in nx.DiGraph
│         Increment node.darkness on each mention
│         Detect relationships, create co-occurrence edges
│         Return: active_node_ids[]
│
├── 2. graph.decay(turn_N)
│         For every node: darkness *= DARKNESS_DECAY (0.95)
│         Decisions: clamped at min 0.5 (never fade completely)
│         Prune nodes below DARKNESS_THRESHOLD (0.1)
│
├── 3. router.update(graph, active_nodes, text=user_input)
│         detect_domain_from_text(text) — keyword-based domain probe
│         signal = encode_graph_to_signal(graph, active_nodes)
│         state  = state × DECAY_RATE + signal × UPDATE_RATE
│         turn_count += 1
│         current_domain = get_domain()   (threshold > 0.01)
│
├── 4. retriever.retrieve_all(graph, router, ...)
│         active_dims = top-20 dims from router.state
│         patterns    = Jaccard similarity scan of Redis patterns
│         gaps        = domain seed common_gaps − active graph nodes
│         contradictions = seed common_contradictions ∩ graph
│
├── 5. prompt_builder.build(...)  ← REWRITTEN (Bug 1 fix)
│         ── PROTECTED BLOCK (guaranteed to reach model) ──
│         session_header + universal_adaptive_block(turn_N)
│         + decisions_section + compressed_history + current_request
│         Cost: ~15 tok/turn for history (was ~300 tok/turn)
│
│         ── OPTIONAL BLOCK (trimmed to fit budget) ──
│         active_context + gaps_section
│
├── 6. inference_config.format_prompt(plugin.system_prompt, prompt, "")
│         ChatML format: <|im_start|>system ... <|im_end|>
│         Note: passes "" not user_input (Bug 2 fix — no double injection)
│
├── 7. cloud_client.generate(prompt, decision, max_tokens)
│         _cloudflare_coding() → POST {"prompt": "<ChatML>", ...}
│         Response parsed from "text" key
│
├── 8. graph.ingest(response_text, turn_N)  ← learn from response
│
├── 9. if confidence > threshold:
│         retriever.store_pattern(active_dims, next_concepts, domain)
│
└── 10. memory.save_session(...)
          Redis pipeline: serialize graph + router + history + metrics
```

---

## File-by-File Reference

---

### brain/config.py

**Purpose:** Single source of truth for all tunable parameters. Every constant is read from `.env` via `python-dotenv`, with a hardcoded fallback.

**Key dataclasses:**

| Dataclass | Fields | Notes |
|-----------|--------|-------|
| `ModelConfig` | `model_path`, `n_ctx`, `n_threads`, `n_batch`, `use_mlock`, `use_mmap`, `temperature`, `max_tokens`, `repeat_penalty` | `n_batch`, `use_mlock`, `use_mmap` added in Optimisation Pass 2 |
| `GraphConfig` | `darkness_decay`, `darkness_increment`, `darkness_threshold`, `max_context_tokens`, etc. | Controls node lifecycle |
| `RouterConfig` | `state_dim`, `decay_rate`, `update_rate`, `top_k_concepts`, `confidence_threshold` | Controls state vector |
| `PromptConfig` | `max_tokens`, `graph_tokens`, `history_tokens`, `routing_tokens` | Budget enforcer limits |
| `BrainConfig` | `max_history_turns`, `domain_detection_threshold` | How much raw history is visible |

A global singleton `config = AppConfig()` is imported by all other modules.

---

### brain/graph.py

**Purpose:** Maintains a directed graph (NetworkX `DiGraph`) of concepts extracted from the conversation. Nodes decay over time; recently mentioned nodes are brighter.

#### Entity Extraction — 4 Layers

```
Text input
   │
   ├── Layer 1: SPRING_CLASSES_REGEX
   │     Matches CamelCase Spring class suffixes:
   │     Template, Service, Repository, Controller, Manager,
   │     Factory, Handler, Mapper, Configuration, Exception, ...
   │
   ├── Layer 2: SPRING_ANNOTATIONS_REGEX
   │     Matches @Annotations explicitly:
   │     @Transactional, @RestController, @Valid, @GetMapping, ...
   │
   ├── Layer 3: Domain keyword lists  (exact-match substring scan)
   │     JDBC_SPECIFIC: NamedParameterJdbcTemplate, RowMapper, ...
   │     SECURITY_SPECIFIC: SecurityFilterChain, JwtAuthFilter, ...
   │     SPRING_EXCEPTION_PATTERNS (v0.2):
   │       GlobalExceptionHandler, ResponseEntity, HttpStatus,
   │       @Valid, @NotBlank, @Email, ProblemDetail, ...
   │
   └── Layer 4: spaCy NER (en_core_web_sm)
         Catches PERSON, ORG, PRODUCT names not caught by regex
```

#### Node Lifecycle

Each node carries a `NodeData` struct:
```
NodeData:
  id                  unique string (e.g. "userdao")
  normalized          lowercase version used for matching
  darkness            float in [0.0, 1.0] — recency weight
  last_active_turn    last turn this node was mentioned
  activation_count    total times mentioned across session
  is_decision         True if marked as a settled architectural decision
```

**Darkness update per mention:**
```
node.darkness = min(1.0, node.darkness + DARKNESS_INCREMENT)
```

**Darkness decay per turn (applied to ALL nodes):**
```
node.darkness *= DARKNESS_DECAY     # e.g. 0.95
if node.darkness < DARKNESS_THRESHOLD:
    remove node from graph
```

Over N turns of silence, a node reaches the threshold at:
```
turns_to_fade = log(THRESHOLD / initial_darkness) / log(DECAY)
             = log(0.1 / 1.0) / log(0.95)  ≈  45 turns
```

#### Relationship Extraction

Relationships are detected via keyword heuristics on the raw text:
- `requires`, `depends on`, `uses` → edge type `"requires"`
- `extends`, `implements`, `inherits` → edge type `"implements"`
- `contradicts`, `conflicts with` → edge type `"contradicts"`

Each edge also carries a `weight` that decays independently.

#### Context Summarisation

`get_context_summary(active_nodes)` produces the structured block injected into the prompt:
```
[GRAPH SUMMARY]
NamedParameterJdbcTemplate [darkness=0.82] → requires → SqlParameterSource
UserDao [darkness=0.71] → implements → JdbcDaoSupport
...
```

---

### brain/router/decay_router.py

**Purpose:** Encodes the current graph state as a 256-dimensional float vector and tracks it through time with geometric decay. This is the "memory" of what the session has been about.

#### State Vector Layout (256 dims)

```
┌──────────────────────────────────────────────────────────┐
│  0  – 63   │ Entity slots  — hashed node positions        │
│  64 – 127  │ Relationship slots — hashed edge types       │
│  128 – 191 │ Domain blocks (one-hot by domain)            │
│             │   [128-135] spring_jdbc                      │
│             │   [136-143] spring_boot                      │
│             │   [144-151] spring_security                  │
│             │   [152-159] spring_async                     │
│             │   [160-167] java_general                     │
│  192 – 255 │ Reserved / future                            │
└──────────────────────────────────────────────────────────┘
```

#### State Update — Geometric Decay

```python
signal = encode_graph_to_signal(graph, active_nodes)   # new turn's contribution
state  = state × DECAY_RATE + signal × UPDATE_RATE
```

With `DECAY_RATE=0.9` and `UPDATE_RATE=0.1`, the contribution of a signal from N turns ago is:
```
weight(N) = UPDATE_RATE × DECAY_RATE^N
```
A signal from 20 turns ago has weight `0.1 × 0.9^20 ≈ 0.012` — negligible.

#### Concept Scoring

To rank which graph nodes are most relevant to the current state:
```python
node_signal = zero vector with 1.0 at node's hashed dim
score = dot(state, node_signal)
```
Higher dot product = node is more "aligned" with the accumulated session signal.

#### Domain Detection

The domain is identified by comparing the mean activation of each domain block against a threshold (0.05). The domain with highest mean wins.

---

### brain/router/retriever.py

**Purpose:** Connects the current session state to stored knowledge in Redis. No ML embeddings — uses Jaccard set similarity on active state dimensions.

#### Jaccard Similarity

```python
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Where `A` and `B` are sets of the **top-20 active dimension indices** from two router state vectors. Two sessions processing similar code will activate the same hash buckets, yielding high Jaccard similarity.

#### Pattern Store / Retrieve Flow

```
High-confidence turn (confidence > threshold):
  active_dims = top-20 active dims from router.state
  pattern_hash = hash(sorted(active_dims))
  Redis key: brain:lfm:patterns:{hash}
  Store: {active_dims, next_concepts, domain, frequency, last_seen}

Later retrieval:
  Scan all brain:lfm:patterns:* keys in Redis
  Compute Jaccard(current_dims, stored_dims)
  Return top-K above SIMILARITY_THRESHOLD
```

#### Gap & Contradiction Detection

Gaps and contradictions are seeded from domain JSON files. At runtime:
- **Gap**: a `common_gap` concept from the domain seed is NOT yet present as a node in the graph
- **Contradiction**: both elements of a `common_contradiction` pair ARE present in the graph

These surfaces in the prompt as `[GAPS DETECTED]` and `[CONTRADICTIONS DETECTED]` sections, giving the LLM proactive guidance.

---

### brain/prompt.py

**Purpose:** Assembles the LLM prompt from all available context signals within a hard token budget.

**Current version:** v0.3 (full rewrite — Bug 1 fix)

#### Two-block assembly model

```
── PROTECTED BLOCK (assembled first, never truncated) ─────────────────
session_header       — domain, turn number, confidence
universal_block      — expert identity + domain detection guide + 5 rules
decisions_section    — graph nodes where is_decision=True and darkness>0.5
compressed_history   — last N turns at ~15 tokens/turn (was ~300 tok/turn)
current_request      — the user's actual question

── OPTIONAL BLOCK (trimmed to fit remaining budget) ───────────────────
active_context       — graph concepts, relationships, recent progress
gaps_section         — domain gaps and contradictions
```

**Why this matters:** The old ordering put history LAST. By turn 4, each assistant response was ~300 tokens. The budget was exhausted before history was appended — the model had no memory of prior turns and rewrote everything from scratch. Compressed history (~15 tokens/turn) means 10 turns costs 150 tokens total.

#### Universal Adaptive Block

`_build_universal_adaptive_block(turn_number)` fires every single turn, bypassing the plugin layer entirely. It gives the model:

1. A domain detection guide (20 domain categories from Java to Creative Writing)
2. 5 universal rules: incremental only, honour decisions, quality floor by domain, no preamble, no questions

This is why consistency = 3.00/3.00 across all three enterprise suites.

#### Decisions are owned exclusively by PromptBuilder

`_build_decisions_section(graph)` emits `[ESTABLISHED DECISIONS]`. This section was also emitted by `graph.get_context_summary()` — Bug 3 fix removed it from the graph side. `get_context_summary()` now only emits `[ACTIVE CONCEPTS]`, `[RELATIONSHIPS]`, `[RECENT PROGRESS]`.

#### Token Budget Enforcement

```python
remaining = get_token_budget(domain) - estimate_tokens(protected)
optional  = active_context + gaps_section

if estimate_tokens(optional) <= remaining:
    body = optional
else:
    body = optional[:remaining * 4]   # char-level trim

return protected + body
```

Domain token budgets (from `AppConfig.TOKEN_BUDGETS`):
- `default`: 2800
- `java_spring`, `spring_boot`, `spring_security`, `spring_jdbc`: 3500
- `sap`: 4000

---

### brain/memory.py

**Purpose:** Persists and restores the full session state (graph, router, history, metrics) to Redis using a single pipeline — all 4 writes are batched into one round-trip.

#### Session Keys

```
brain:session:{session_id}:graph     → serialized ContextGraph JSON
brain:session:{session_id}:router    → serialized DecayRouter JSON
brain:session:{session_id}:history   → JSON array of {input, response, turn}
brain:session:{session_id}:metrics   → JSON array of per-turn metric dicts
```

All keys carry TTL = `REDIS_TTL_SESSION` (default 24 hours).

#### Serialization

- `ContextGraph.serialize()` → converts `nx.DiGraph` nodes/edges to plain dict lists + episode list → JSON string
- `DecayRouter.serialize()` → converts `np.ndarray` state to Python list → JSON string
- Both have matching `.deserialize(json_str)` class methods

---

### brain/brain.py

**Purpose:** The orchestrator. `Brain.think(user_input)` runs the full pipeline in sequence and returns `(response_text, turn_metrics)`.

The class is initialized once per session with a `user_id` and `session_id`. It loads any existing session from Redis and initialises the LLM with all CPU throughput flags.

**Per-turn metrics emitted:**

```python
{
  "turn":                 int,
  "prompt_tokens":        int,      # chars // 4 estimate
  "response_tokens":      int,
  "response_time_ms":     float,
  "tokens_per_second":    float,
  "graph_nodes":          int,
  "graph_edges":          int,
  "routing_confidence":   float,    # L2 norm of router state
  "domain":               str,
  "gaps_count":           int,
  "contradictions_count": int,
  "redis_retrieval_ms":   float,
  "graph_ops_ms":         float
}
```

---

### seed.py & seed/*.json

**Purpose:** One-shot loader that reads domain JSON files and writes them to Redis as domain knowledge seeds. Run once before any benchmark or production session.

#### Seed JSON Schema

```json
{
  "domain": "spring_jdbc",
  "trigger_concepts": ["NamedParameterJdbcTemplate", "RowMapper", ...],
  "typical_flow": ["DataSource config", "DAO class", ...],
  "common_gaps": ["error handling", "connection pooling", ...],
  "common_contradictions": [
    ["JdbcTemplate", "NamedParameterJdbcTemplate"]
  ]
}
```

The seeder writes each domain to:
```
brain:lfm:domain:{domain_name}:knowledge → JSON blob
```

Three domains ship by default: `spring_jdbc`, `spring_boot`, `spring_security`.

---

### tests/quality/scorer.py

**Purpose:** Evaluates Java/Spring Boot code quality on a 0–11 rubric.

| Dimension | Max | How Measured |
|-----------|----:|--------------|
| Compilability | 1.0 | Balanced `{}`, `public class` has closing brace; strips markdown fences before analysis |
| Correctness   | 3.0 | `required_keywords` from benchmark turn config, weighted |
| Consistency   | 3.0 | Penalises `@Autowired` field injection, wrong JDBC type |
| Completeness  | 2.0 | Error handling: `DataAccessException`, `@Transactional`, `EmptyResultDataAccessException`; edge cases: `orElseThrow`, `Optional.ofNullable`, `UserNotFoundException` |
| Convention    | 2.0 | Exception layer (+1.0), validation (`@Valid`, `@NotBlank`) (+0.5), REST responses (`ResponseEntity`) (+0.5) |

### tests/quality/generic_scorer.py  *(new)*

**Purpose:** Language-agnostic scorer for non-Java suites. Same 11-point breakdown.

| Language | Completeness patterns | Convention patterns |
|---|---|---|
| `python` | `IntegrityError`, `NoResultFound`, `SQLAlchemyError`, `HTTPException`, `async with`, HTTP status codes | `async def`, `await`, `Depends(`, type hints `-> `, `db.rollback()`, `Field(` |
| `go`     | `if err != nil`, `pgx.ErrNoRows`, `defer `, `rows.Close()`, `tx.Rollback`, HTTP status consts | `ctx context.Context`, `interface {`, `func (`, `pgx.BeginTx(`, `defer tx.Rollback`, `json:"` |
| `generic`| `error`, `exception`, `null`, `nil`, `catch`, `try` | `function`, `return`, `const`, `interface` |

Compilability for Python checks that every `def`/`class`/`async def` block has an indented body. Compilability for Go checks balanced `{}`.

Run enterprise suites with:
```bash
PYTHONPATH=$PWD python3 benchmarks/enterprise_runner.py --suite python
PYTHONPATH=$PWD python3 benchmarks/enterprise_runner.py --suite go
PYTHONPATH=$PWD python3 benchmarks/enterprise_runner.py --all
```

---

### tests/quality/coherence.py

**Purpose:** Measures how well the model's response reflects what the graph says is currently active.

#### v0.2 — Lemmatised CoherenceChecker

```python
def _concept_present(concept, response_lemmas, response_lower):
    # Fast path: exact substring (works for CamelCase class names)
    if concept in response_lower:
        return True
    # Lemma path: all content words of concept appear in response lemmas
    words = concept.split()
    if len(words) > 1:
        significant = [w for w in words if len(w) > 2]
        return all(w in response_lemmas for w in significant)
    return concept in response_lemmas
```

**v0.1 bug:** used `node.normalized in response.lower()` — exact match only. "constructor injection" failed when response said "constructor-injected", producing 0.00 coherence on turn 2.

**v0.2 fix:** spaCy lemmatisation before comparison. "constructor-injected" → lemma "inject" matches "inject" from "injection".

---

### benchmarks/runner.py

**Purpose:** Single-suite Brain-only benchmark. Baseline was local-GGUF only and is skipped in cloud mode.

#### Brain Strategy

`run_brain(suite)` uses `Brain.process(turn["input"])` directly. `Brain.process()` is a wrapper around `Brain.think()`. Metrics are read from `brain.last_prompt_tokens`, `brain.last_graph_ms`, `brain.last_retrieval_ms`.

#### Output

CSV written to `data/results/brain/dao_suite.csv`.

Run:
```bash
PYTHONPATH=$PWD python3 benchmarks/runner.py
```

### benchmarks/enterprise_runner.py  *(new)*

**Purpose:** Multi-suite enterprise runner that can run all three language suites and print a cross-suite comparison.

#### Benchmark suites

| Key | File | Language | Description |
|---|---|---|---|
| `java` | `dao_suite.json` | Java | Original Spring Boot DAO, 10 turns |
| `python` | `python_inventory_suite.json` | Python | FastAPI + async SQLAlchemy inventory system, 10 turns |
| `go` | `go_ledger_suite.json` | Go | Financial ledger API with pgx/v5 + chi, 10 turns |

Each turn in the JSON carries:
```json
{
  "number": 3,
  "input": "...",
  "quality_config": {
    "language": "python",
    "required_keywords": {"keyword": weight, ...}
  }
}
```

The runner selects `SpringBootQualityScorer` for Java and `GenericCodeQualityScorer(language)` for all others.

---

## Optimisation Pass 3 — Cloud Migration + Prompt Architecture

These changes were applied on **February 26–28, 2026** and represent the current production state.

### Cloud migration (Feb 26)

| Change | Effect |
|---|---|
| Removed `from llama_cpp import Llama` entirely | No local model instantiation |
| `self.llm = None` always | All inference routes to Cloudflare |
| `inference_config = MODEL_REGISTRY["qwen2.5-coder-32b"]` hardcoded | ChatML format always used |
| `_cloudflare_coding()` switched from `messages` to `prompt` field | Worker was silently ignoring `messages` — quality was 4.05 |
| Response parsed from `"text"` key | Worker response shape: `{ok, model, prompt, text, raw}` |

**Quality: 4.05 → 9.60/11** from the `messages→prompt` fix alone.

### 5-bug fix (Feb 28)

| Bug | File | Fix | Effect |
|---|---|---|---|
| History always truncated + no universal domain support | `brain/prompt.py` | Full rewrite: protected block, compressed history, universal adaptive block | Context preserved every turn; any domain works |
| `user_input` injected twice | `brain/brain.py` | Pass `""` not `user_input` to `format_prompt()` | Removes duplicate request tokens |
| Decisions emitted twice | `brain/graph.py` | Remove decisions block from `get_context_summary()` | Single authoritative decisions section |
| Missing Bean Validation mandates | `brain/plugins/java_spring.py` | Added `@Valid`, `@NotBlank`, `@Email`, `@Size`, `@Transactional` rules | Turn 9 quality: 6.5 → 11.0 |
| Plugin personas too thin | `brain/plugins/general.py`, `sap.py` | Session rules + domain standards | Consistent incremental behaviour |

**Java quality: 9.60 → 9.75/11 | Perfect turns: 2 → 5 | Avg prompt tokens: 1,838 → 1,712**

---

## Optimisation Pass 1 — Quality (historical)

These changes improved **brain avg quality from 5.4 → 6.5/11** and **coherence from 0.51 → 0.79**.

### 1. Raise `PROMPT_MAX_TOKENS` 2800 → 3500

**Why:** Brain prompt was hitting the budget cap at turn 4. The prompt builder was silently truncating the graph context summary and recent history before the LLM could see it. Turns 7–10 were generating with incomplete context.

**Effect:** Turn 7 quality rose from 3.0 → 7.0/11. Turn 8 from 3.0 → 10.0/11.

**Tradeoff:** Larger budget = fewer savings over baseline in token count (2.4× peak vs 4.0× in v0.1). This is the core quality/compression slider in the system.

### 2. Improved System Prompt

**Why:** The old prompt was vague. It said "build incrementally" but did not prohibit rewriting existing classes. Models frequently rewrote the entire `UserDao` when asked only to "add pagination."

**Change:** Added 5 explicit SESSION RULES covering incremental build, decision honour, precision (emit additions only), no questions, code only.

### 3. Quality Scorer — Keyword Calibration

**Why:** Turns 7–10 (`@ControllerAdvice`, `ResponseEntity`, `@Valid`, `@ExceptionHandler`) had sparse or missing keyword sets in `dao_suite.json`, so scores deflated to 3.0/11 regardless of actual output quality.

**Change:** Filled in all keyword sets for all 10 turns. Turn 8 now checks for `@RestController`, `@RequestMapping`, `ResponseEntity`, `@GetMapping`, `@PostMapping`.

### 4. Coherence Checker — Lemmatisation

**Why:** Exact match `node.normalized in response.lower()` failed for phrased variations. Node "constructor injection" → response "constructor-injected" → no match → coherence=0.00 on turn 2.

**Change:** Rewrote with spaCy lemmatisation + two-pass check (substring fast path + lemma bag check). Coherence on turn 2 fixed: 0.00 → 1.00.

### 5. `SPRING_EXCEPTION_PATTERNS` in Entity Extraction

**Why:** Exception-handling concepts (`GlobalExceptionHandler`, `ResponseEntity`, `HttpStatus`, `@ControllerAdvice`) never entered the graph because they were not in the regex or keyword lists. The router could not detect the domain shift in turns 6–10, and the prompt context did not include them.

**Change:** Added `SPRING_EXCEPTION_PATTERNS` list with 23 entries. These are added to the keyword scan in `extract_entities()`.

---

## Optimisation Pass 2 — CPU Throughput (historical, local model only)

These changes improved **baseline avg TPS from 1.8 → 2.3** (+28%) and **brain avg TPS from 0.9 → 1.1** (+22%) without any code logic change. Not applicable in cloud-only mode — included for reference if local inference is re-enabled.

### 1. `n_batch=512`

**What it is:** The number of tokens processed in parallel during prompt evaluation (the KV cache fill phase). This is not the generation batch size — it affects how fast the model reads in the prompt before generating.

**Why it helps:** Default `n_batch=512` is already reasonably tuned, but by making it explicit and configurable, you can tune it per machine. On machines with more L2/L3 cache, higher `n_batch` is faster; on RAM-constrained machines, lower is safer.

### 2. `use_mlock=True`

**What it is:** Instructs the OS to pin the model weights in physical RAM so they cannot be paged out.

**Why it matters:** A 2GB GGUF model competes with the Redis server and Python heap for RAM. Without `mlock`, the OS may page model layers out between turns, causing IO stalls on the next access. With `mlock`, every weight is always hot in RAM.

**Warning:** Requires sufficient RAM. If you have <4GB free, set to `false`.

### 3. `use_mmap=True`

**What it is:** Memory-maps the model file, allowing the OS to handle paging of model weights lazily rather than loading the entire file at startup.

**Why it matters:** First load is faster. If combined with `use_mlock`, the weights are loaded once and then pinned — best of both worlds.

### 4. Temperature 0.7 → 0.2

**Why it affects speed:** Lower temperature computation is marginally faster (fewer random samples needed for convergence in greedy-adjacent sampling). More importantly, at temperature 0.2 the model produces shorter, denser code responses — less generation time, higher tokens/sec ratio because the generated tokens per call is smaller.

**Quality effect:** Lower temperature = more deterministic. For code generation this is generally desirable (less hallucination of non-existent APIs).

### 5. `n_threads` — Physical Cores Only

**Why:** llama.cpp uses BLAS for matrix multiplication. Hyperthreaded logical cores share execution units and memory bandwidth. Setting `n_threads` to physical core count (not logical) prevents contention. Use `lscpu | grep "Core(s) per socket"` to find the right value for your machine.

---

## Configuration Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MODEL_N_CTX` | `8192` | Context window — must be ≥ longest prompt you'll send |
| `MODEL_N_THREADS` | `4` | **Set to physical core count** — not logical (hyperthreading hurts) |
| `MODEL_N_BATCH` | `512` | Prompt evaluation parallelism — higher = faster fill, more RAM |
| `MODEL_USE_MLOCK` | `true` | Pin model in RAM — disable if RAM constrained |
| `MODEL_USE_MMAP` | `true` | Memory-map model file |
| `MODEL_TEMPERATURE` | `0.2` | Lower = more deterministic code, faster generation |
| `PROMPT_MAX_TOKENS` | `3500` | **Primary quality/compression lever** — higher = better quality, less compression vs baseline |
| `GRAPH_DARKNESS_DECAY` | `0.95` | Per-turn node fade rate — lower fades faster |
| `GRAPH_DARKNESS_INCREMENT` | `0.3` | How much a mention brightens a node |
| `ROUTER_DECAY_RATE` | `0.9` | State vector decay — lower = shorter memory |
| `ROUTER_UPDATE_RATE` | `0.1` | Weight of new signal — must sum to ≤ 1.0 with decay |
| `BRAIN_MAX_HISTORY_TURNS` | `3` | Turns kept in compressed history block (~15 tok/turn each) |
| `GRAPH_MAX_CONTEXT_TOKENS` | `800` | Token budget for graph summary section (was 300) |
| `RETRIEVER_SIMILARITY_THRESHOLD` | `0.3` | Jaccard threshold for pattern retrieval (was 0.7) |
| `RETRIEVER_MIN_PATTERN_FREQUENCY` | `1` | Minimum pattern hits before retrieval (was 2) |
| `CLOUDFLARE_WORKER_URL` | *(set in .env)* | Inference endpoint — if set, all routing goes to Cloudflare |

---

## Redis Key Schema

```
brain:session:{session_id}:graph        → ContextGraph JSON (TTL: 24h)
brain:session:{session_id}:router       → DecayRouter JSON (TTL: 24h)
brain:session:{session_id}:history      → History array JSON (TTL: 24h)
brain:session:{session_id}:metrics      → Metrics array JSON (TTL: 24h)

brain:lfm:domain:{domain}:knowledge     → Domain seed JSON (no TTL, set by seed.py)
brain:lfm:patterns:{hash}               → Pattern JSON (TTL: 30 days)
```

Pattern keys use a hash of the sorted top-20 active dimension index tuple, making collision probability negligible for typical session counts.
