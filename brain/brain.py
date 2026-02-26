import time
from typing import Tuple, Dict, Any, List

from llama_cpp import Llama

from brain.config import config
from brain.graph import ContextGraph
from brain.router.decay_router import DecayRouter
from brain.router.retriever import Retriever
from brain.prompt import PromptBuilder
from brain.memory import Memory

class Brain:
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        
        self.memory = Memory()
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        
        # Load session state
        self.graph, self.router, self.history, self.metrics = self.memory.load_session(session_id)
        
        # Initialize LLM with CPU throughput optimisations
        print(f"Loading model from {config.model.model_path}...")
        self.llm = Llama(
            model_path=config.model.model_path,
            n_ctx=config.model.n_ctx,
            n_threads=config.model.n_threads,
            n_batch=config.model.n_batch,       # parallelise prompt KV fill
            use_mlock=config.model.use_mlock,   # pin model in RAM
            use_mmap=config.model.use_mmap,     # memory-map model file
            verbose=False
        )
        
    def think(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        turn_number = self.router.turn_count + 1
        
        # 1. Graph Ingestion & Decay
        graph_start = time.time()
        active_nodes = self.graph.ingest(user_input, turn_number)
        self.graph.decay(turn_number)
        graph_ms = (time.time() - graph_start) * 1000
        
        # 2. Router Update
        self.router.update(self.graph, active_nodes)
        
        # 3. Retrieval
        retrieval_start = time.time()
        retrieved_data = self.retriever.retrieve_all(
            self.graph, self.router, self.user_id, self.session_id
        )
        retrieval_ms = (time.time() - retrieval_start) * 1000
        
        # 4. Prompt Building — pass history list (not graph.episodes which includes response text)
        prompt = self.prompt_builder.build(
            user_input=user_input,
            graph=self.graph,
            router=self.router,
            retrieved_data=retrieved_data,
            history=self.history,
            turn_number=turn_number
        )
        prompt_tokens = len(prompt) // 4  # rough estimate
        
        # 5. Model Generation
        generation_start = time.time()
        response = self.llm(
            prompt,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            repeat_penalty=config.model.repeat_penalty,
            stop=["[USER]", "[CURRENT REQUEST]"]
        )
        response_text = response["choices"][0]["text"].strip()
        generation_time = time.time() - generation_start
        response_tokens = len(response_text) // 4
        
        tps = response_tokens / generation_time if generation_time > 0 else 0
        
        # 6. Metrics & Memory Update
        turn_metrics = {
            "turn": turn_number,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_time_ms": generation_time * 1000,
            "tokens_per_second": tps,
            "graph_nodes": len(self.graph.nx_graph.nodes),
            "graph_edges": len(self.graph.nx_graph.edges),
            "routing_confidence": self.router.get_confidence(),
            "domain": self.router.current_domain,
            "gaps_count": len(retrieved_data.get("gaps", [])),
            "contradictions_count": len(retrieved_data.get("contradictions", [])),
            "redis_retrieval_ms": retrieval_ms,
            "graph_ops_ms": graph_ms
        }
        
        self.metrics.append(turn_metrics)
        self.history.append({"input": user_input, "response": response_text, "turn": turn_number})
        
        # Ingest the RESPONSE into the graph so decisions/concepts from the answer are tracked
        # We do NOT increment turn_count here — it was already incremented by router.update()
        self.graph.ingest(response_text, turn_number)
        
        # Store patterns if confidence is high
        if self.router.get_confidence() > config.router.confidence_threshold:
            active_dims = self.retriever.get_top_active_dims(self.router.state.tolist())
            next_concepts = self.router.get_active_concepts(self.graph)
            self.retriever.store_pattern(active_dims, next_concepts, self.router.current_domain)
            
        # Save session
        self.memory.save_session(self.session_id, self.graph, self.router, self.history, self.metrics)
        
        return response_text, turn_metrics
