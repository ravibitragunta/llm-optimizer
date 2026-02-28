# LLM Brain — Benchmark Results
**Last updated:** February 28, 2026  
**Inference backend:** Cloudflare Workers — `qwen-2.5-coder-32b`  
**Mode:** Cloud-only (`self.llm = None`, no local GGUF)

---

## Current Results: 3-Suite Enterprise Benchmark

### Suite 1 — Java Spring Boot DAO
**Avg Quality: 9.75/11 | Perfect turns: 5/10**

| Turn | Prompt Tokens | TPS  | Quality  | Coherence | Graph Nodes |
|------|---------------|------|----------|-----------|-------------|
| 1    | 1522          | 34.2 | 10.5/11  | 1.00      | 30          |
| 2    | 1572          | 31.6 |  9.0/11  | 1.00      | 34          |
| 3    | 1641          | 33.3 |  8.0/11  | 1.00      | 48          |
| 4    | 1677          | 30.4 |  9.5/11  | 0.27      | 56          |
| 5    | 1748          | 10.5 |  8.0/11  | 0.18      | 61          |
| 6    | 1797          | 34.3 | 10.5/11  | 0.42      | 75          |
| 7    | 1773          | 20.9 |  9.0/11  | 0.00      | 80          |
| 8    | 1796          | 40.8 | 11.0/11  | 0.17      | 104         |
| 9    | 1784          | 34.5 | 11.0/11  | 0.50      | 124         |
| 10   | 1806          | 20.9 | 11.0/11  | 0.17      | 142         |
| **Avg** | **1712** | **29.1** | **9.75** | **0.47** | |

**Dimension breakdown:**

| Dimension | Avg | Max |
|---|---|---|
| Compilability | 1.00 | 1.0 |
| Correctness   | 3.00 | 3.0 |
| Consistency   | 3.00 | 3.0 |
| Completeness  | 1.70 | 2.0 |
| Convention    | 1.05 | 2.0 |

---

### Suite 2 — Python FastAPI Inventory Management
**10 turns — async SQLAlchemy 2.0, PostgreSQL, Alembic, pytest-asyncio**  
**Avg Quality: 8.64/11 | Perfect turns: 1/10**

| Turn | Prompt Tokens | TPS  | Quality  | Coherence | Graph Nodes |
|------|---------------|------|----------|-----------|-------------|
| 1    | 1628          | 28.9 |  7.3/11  | 1.00      | 39          |
| 2    | 1602          | 34.4 | 11.0/11  | 1.00      | 70          |
| 3    | 1717          | 36.1 |  9.5/11  | 1.00      | 98          |
| 4    | 1815          | 34.4 |  8.8/11  | 0.88      | 126         |
| 5    | 1815          | 17.1 |  9.1/11  | 0.78      | 148         |
| 6    | 1869          | 28.1 |  9.9/11  | 0.60      | 178         |
| 7    | 1816          | 36.1 |  9.0/11  | 0.23      | 210         |
| 8    | 1861          | 26.7 |  5.5/11  | 0.18      | 243         |
| 9    | 1843          | 39.8 |  9.0/11  | 0.50      | 269         |
| 10   | 1882          | 17.7 |  7.3/11  | 0.62      | 305         |
| **Avg** | **1785** | **29.9** | **8.64** | **0.68** | |

**Dimension breakdown:**

| Dimension | Avg | Max |
|---|---|---|
| Compilability | 1.00 | 1.0 |
| Correctness   | 2.30 | 3.0 |
| Consistency   | 3.00 | 3.0 |
| Completeness  | 1.20 | 2.0 |
| Convention    | 1.14 | 2.0 |

> Turn 8 (Alembic migration DDL) scored 5.5/11 — migrations carry no async/error patterns so completeness and convention floor out. All other turns avg 9.2/11.

---

### Suite 3 — Go Financial Ledger API
**10 turns — pgx/v5, chi router, cursor pagination, serializable transactions, slog**  
**Avg Quality: 7.40/11 | Perfect turns: 0/10**

| Turn | Prompt Tokens | TPS  | Quality  | Coherence | Graph Nodes |
|------|---------------|------|----------|-----------|-------------|
| 1    | 1615          | 27.5 |  6.5/11  | 1.00      | 40          |
| 2    | 1655          | 29.6 |  8.0/11  | 1.00      | 71          |
| 3    | 1688          | 30.6 |  9.0/11  | 0.67      | 103         |
| 4    | 1715          | 30.8 |  8.5/11  | 0.83      | 128         |
| 5    | 1787          | 15.2 |  5.3/11  | 0.43      | 162         |
| 6    | 1822          | 23.2 |  8.5/11  | 0.44      | 189         |
| 7    | 1835          | 33.0 |  7.7/11  | 0.55      | 215         |
| 8    | 1826          | 29.6 |  6.5/11  | 0.58      | 246         |
| 9    | 1798          | 23.7 |  9.0/11  | 0.42      | 278         |
| 10   | 1885          | 12.6 |  5.0/11  | 0.00      | 315         |
| **Avg** | **1763** | **25.6** | **7.40** | **0.59** | |

**Dimension breakdown:**

| Dimension | Avg | Max |
|---|---|---|
| Compilability | 1.00 | 1.0 |
| Correctness   | 1.85 | 3.0 |
| Consistency   | 3.00 | 3.0 |
| Completeness  | 0.50 | 2.0 |
| Convention    | 1.05 | 2.0 |

> Correctness gap: model inconsistently emits precise pgx/v5 API names (`pgx.TxOptions`, `TxIsoLevelSerializable`, `ErrUnbalancedEntries`). Completeness gap: exhaustive `if err != nil` + `defer tx.Rollback` coverage degrades after turn 5.

---

## Cross-Suite Summary

| Suite | Avg Quality | Coherence | Compilability | Correctness | Consistency | Completeness | Convention |
|---|---|---|---|---|---|---|---|
| Java Spring Boot | **9.75/11** | 0.47 | 1.00 | 3.00 | 3.00 | 1.70 | 1.05 |
| Python FastAPI   | **8.64/11** | 0.68 | 1.00 | 2.30 | 3.00 | 1.20 | 1.14 |
| Go Ledger API    | **7.40/11** | 0.59 | 1.00 | 1.85 | 3.00 | 0.50 | 1.05 |

**Consistency = 3.00/3.00 across all suites.** The universal adaptive block + protected decisions section prevents the model from ever contradicting session decisions.

---

## Historical Quality Progression

| Date | Suite | Avg Quality | Key Change |
|---|---|---|---|
| Feb 26 | Java DAO | 4.05/11 | Cloudflare migration — worker ignored `messages` field |
| Feb 26 | Java DAO | 9.60/11 | Fixed: switched to `prompt` + inline ChatML format |
| Feb 28 | Java DAO | **9.75/11** | 5-bug fix: compressed history, universal block, dedup decisions |
| Feb 28 | Python FastAPI | **8.64/11** | New enterprise suite |
| Feb 28 | Go Ledger | **7.40/11** | New enterprise suite |

---

## Infrastructure

| Setting | Value |
|---|---|
| Worker URL | `https://qwen-coder-worker.ai-model-pulse.workers.dev` |
| Model | `@cf/qwen/qwen2.5-coder-32b-instruct` |
| Request format | `{"model": "qwen-2.5-coder-32b", "prompt": "<ChatML>", "max_tokens": N}` |
| Response key | `"text"` in `{ok, model, prompt, text, raw}` |
| Auth | None |
| Summarisation | Same worker, same model, every 5 turns |
| Summarisation status | ⚠️ Returns truncated JSON — caught gracefully, returns `{}` |

---

## Known Ceilings

| Issue | Impact | Fix |
|---|---|---|
| Summarisation truncated | Graph decisions stale after turn 5 | Simpler user-message-only JSON schema |
| Go completeness low (0.50) | Go avg 7.40 vs Java 9.75 | Inject exact pgx/v5 API names into Go plugin system prompt |
| Alembic migration score (5.5/11) | Pulls Python avg below 9.0 | Add migration-specific completeness patterns to scorer |
| Coherence collapse after turn 4 | 150–300 nodes accumulated | Lower `GRAPH_DARKNESS_INCREMENT` 0.3→0.2, cap at 80 active nodes |
| Convention ceiling (avg ~1.05 of 2.0) | ~1 pt lost per turn | Add per-language convention patterns to `GenericCodeQualityScorer` |
