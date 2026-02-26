import json
import time
import os
import csv
import gc

from llama_cpp import Llama
from brain.config import config
from brain.graph import ContextGraph
from brain.router.decay_router import DecayRouter
from brain.router.retriever import Retriever
from brain.prompt import PromptBuilder
from brain.memory import Memory
from brain.brain import Brain
from tests.quality.scorer import SpringBootQualityScorer
from tests.quality.coherence import CoherenceChecker


def load_suite(path="benchmarks/conversations/dao_suite.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_baseline(suite, llm):
    """Vanilla Qwen — full accumulated history each turn (exponential context growth)."""
    print("\n=== BASELINE (Vanilla Qwen) ===")
    system_prompt = (
        "You are a senior Spring Boot engineer. "
        "Generate production-quality Java code.\n\n"
    )
    history = ""
    results = []

    for turn in suite:
        prompt = system_prompt + history + "[USER]\n" + turn["input"] + "\n[ASSISTANT]\n"
        prompt_tokens = len(prompt) // 4

        start = time.time()
        try:
            response = llm(
                prompt,
                max_tokens=config.model.max_tokens,
                temperature=config.model.temperature,
                repeat_penalty=config.model.repeat_penalty,
                stop=["[USER]"],
            )
        except ValueError as e:
            # Context window exceeded — truncate oldest history and retry once
            # This is expected late in a long session (proves context explosion)
            words = history.split("[USER]")[-3:]  # keep only last 3 exchanges
            history = "[USER]".join(words)
            prompt = system_prompt + history + "[USER]\n" + turn["input"] + "\n[ASSISTANT]\n"
            try:
                response = llm(prompt, max_tokens=config.model.max_tokens, temperature=config.model.temperature, repeat_penalty=config.model.repeat_penalty, stop=["[USER]"])
            except ValueError:
                response = {"choices": [{"text": "[CONTEXT OVERFLOW — TRUNCATED]"}]}
        elapsed = time.time() - start

        response_text = response["choices"][0]["text"].strip()
        resp_tokens = len(response_text) // 4
        tps = resp_tokens / elapsed if elapsed > 0 else 0

        history += f"[USER]\n{turn['input']}\n[ASSISTANT]\n{response_text}\n"

        rec = {
            "turn": turn["number"],
            "prompt_tokens": prompt_tokens,
            "response_tokens": resp_tokens,
            "tps": tps,
            "response": response_text,
        }
        results.append(rec)
        print(
            f"  Turn {turn['number']:2d}: prompt_tokens={prompt_tokens:5d}  tps={tps:5.1f}"
        )

    return results


def run_brain(suite, llm):
    """Brain — context graph + decay router (bounded context)."""
    print("\n=== BRAIN (Context Graph + Decay Router) ===")

    # Fresh session for each benchmark run — flush any stale Redis state
    session_id = f"dao_bench_{int(time.time())}"
    memory = Memory()
    graph = ContextGraph()
    router = DecayRouter()
    retriever = Retriever()
    prompt_builder = PromptBuilder()
    scorer = SpringBootQualityScorer()
    coherence_checker = CoherenceChecker()

    history: list = []
    metrics_list: list = []
    results = []

    for turn in suite:
        turn_number = router.turn_count + 1

        # 1. Graph ingest + decay
        t0 = time.time()
        active_nodes = graph.ingest(turn["input"], turn_number)
        graph.decay(turn_number)
        graph_ms = (time.time() - t0) * 1000

        # 2. Router update
        router.update(graph, active_nodes)

        # 3. Retrieval
        t0 = time.time()
        retrieved = retriever.retrieve_all(graph, router, "bench_user", session_id)
        retrieval_ms = (time.time() - t0) * 1000

        # 4. Prompt build
        prompt = prompt_builder.build(
            user_input=turn["input"],
            graph=graph,
            router=router,
            retrieved_data=retrieved,
            history=history,
            turn_number=turn_number,
        )
        prompt_tokens = len(prompt) // 4

        # 5. Generate
        t0 = time.time()
        response = llm(
            prompt,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            repeat_penalty=config.model.repeat_penalty,
            stop=["[CURRENT REQUEST]", "[USER]"],
        )
        elapsed = time.time() - t0

        response_text = response["choices"][0]["text"].strip()
        resp_tokens = len(response_text) // 4
        tps = resp_tokens / elapsed if elapsed > 0 else 0

        # 6. Update state
        history.append({"input": turn["input"], "response": response_text, "turn": turn_number})
        graph.ingest(response_text, turn_number)   # learn from response

        if router.get_confidence() > config.router.confidence_threshold:
            active_dims = retriever.get_top_active_dims(router.state.tolist())
            next_concepts = router.get_active_concepts(graph)
            retriever.store_pattern(active_dims, next_concepts, router.current_domain)

        quality = scorer.score(response_text, turn.get("quality_config", {}), [])
        coh = coherence_checker.check(response_text, graph, turn_number)

        rec = {
            "turn": turn_number,
            "prompt_tokens": prompt_tokens,
            "response_tokens": resp_tokens,
            "tps": tps,
            "quality": quality.total,
            "coherence": coh,
            "graph_nodes": len(graph.nx_graph.nodes),
            "graph_edges": len(graph.nx_graph.edges),
            "domain": router.current_domain,
            "gaps": len(retrieved.get("gaps", [])),
            "graph_ms": round(graph_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
        }
        results.append(rec)
        print(
            f"  Turn {turn_number:2d}: prompt_tokens={prompt_tokens:5d}  tps={tps:5.1f}"
            f"  quality={quality.total:.1f}/11  coherence={coh:.2f}"
            f"  nodes={len(graph.nx_graph.nodes)}"
        )

    return results


def save_csv(results, path):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results → {path}")


def print_comparison(baseline, brain):
    print("\n" + "=" * 75)
    print(f"{'Turn':>4} | {'Base Tokens':>11} | {'Brain Tokens':>12} | {'Reduction':>9} | {'Base TPS':>8} | {'Brain TPS':>9}")
    print("-" * 75)
    for b, br in zip(baseline, brain):
        reduction = b["prompt_tokens"] / br["prompt_tokens"] if br["prompt_tokens"] > 0 else 0
        print(
            f"{b['turn']:>4} | {b['prompt_tokens']:>11} | {br['prompt_tokens']:>12} | {reduction:>8.1f}x | {b['tps']:>8.1f} | {br['tps']:>9.1f}"
        )
    print("=" * 75)

    # Aggregate stats
    avg_base_tokens = sum(r["prompt_tokens"] for r in baseline) / len(baseline)
    avg_brain_tokens = sum(r["prompt_tokens"] for r in brain) / len(brain)
    avg_reduction = avg_base_tokens / avg_brain_tokens if avg_brain_tokens > 0 else 0
    avg_base_tps = sum(r["tps"] for r in baseline) / len(baseline)
    avg_brain_tps = sum(r["tps"] for r in brain) / len(brain)
    avg_quality = sum(r["quality"] for r in brain) / len(brain)
    avg_coherence = sum(r["coherence"] for r in brain) / len(brain)

    peak_base = max(r["prompt_tokens"] for r in baseline)
    peak_brain = max(r["prompt_tokens"] for r in brain)

    print(f"\n{'SUMMARY':}")
    print(f"  Avg prompt tokens   — Baseline: {avg_base_tokens:.0f}   Brain: {avg_brain_tokens:.0f}   ({avg_reduction:.1f}x reduction)")
    print(f"  Peak prompt tokens  — Baseline: {peak_base}   Brain: {peak_brain}")
    print(f"  Avg tokens/sec      — Baseline: {avg_base_tps:.1f}   Brain: {avg_brain_tps:.1f}")
    print(f"  Brain avg quality   — {avg_quality:.1f}/11")
    print(f"  Brain avg coherence — {avg_coherence:.2f}")
    # Quadratic compute reduction estimate (token^2 proxy)
    compute_reduction = (avg_base_tokens ** 2) / (avg_brain_tokens ** 2) if avg_brain_tokens > 0 else 0
    print(f"  Estimated compute reduction (token² ratio): {compute_reduction:.0f}×")


if __name__ == "__main__":
    suite = load_suite()
    print(f"Loaded {len(suite)}-turn DAO suite")

    # Load a SINGLE shared LLM instance — avoids double-loading 2 GB model
    # Use n_ctx=8192 so the baseline can accumulate history across all 10 turns
    # (the baseline WILL hit this limit by turn 6+, proving context explosion)
    print(f"\nLoading model: {config.model.model_path}")
    llm = Llama(
        model_path=config.model.model_path,
        n_ctx=8192,
        n_threads=config.model.n_threads,
        n_batch=config.model.n_batch,
        use_mlock=config.model.use_mlock,
        use_mmap=config.model.use_mmap,
        verbose=False,
    )

    # --- BASELINE ---
    baseline_results = run_baseline(suite, llm)
    save_csv(baseline_results, "data/results/baseline/dao_suite.csv")

    # --- BRAIN ---
    brain_results = run_brain(suite, llm)
    save_csv(brain_results, "data/results/brain/dao_suite.csv")

    # --- COMPARISON ---
    print_comparison(baseline_results, brain_results)
