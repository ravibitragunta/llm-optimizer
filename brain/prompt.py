from typing import List, Dict, Any
from brain.config import config
from brain.graph import ContextGraph
from brain.router.decay_router import DecayRouter

class PromptBuilder:
    def __init__(self):
        self.system_prompt = (
            "You are a senior software engineer generating production-quality code.\n"
            "\n"
            "SESSION RULES (follow strictly):\n"
            "1. INCREMENTAL BUILD: You are continuing a session where code already exists. "
            "Do NOT rewrite or repeat classes/methods already established. "
            "Only output the NEW code asked for.\n"
            "2. HONOUR DECISIONS: All items in [ESTABLISHED DECISIONS] are final. "
            "Never contradict or replace them.\n"
            "3. PRECISION: If asked to add a method or annotation, emit only that addition, "
            "not the entire enclosing class again.\n"
            "4. NO QUESTIONS: Never ask for clarification. "
            "Infer from context and produce the best possible answer.\n"
            "5. CODE ONLY: Output code, configuration, or direct explanation. "
            "No preamble, no meta-commentary.\n"
        )
        
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4
        
    def build(self, 
              user_input: str,
              graph: ContextGraph,
              router: DecayRouter,
              retrieved_data: Dict[str, Any],
              history: List[Dict],
              turn_number: int) -> str:
                  
        domain = router.current_domain
        confidence = f"{router.get_confidence():.2f}"
        
        session_context = f"[SESSION CONTEXT]\nDomain: {domain}\nConfidence: {confidence}\nSession progress: Turn {turn_number} of conversation\n\n"
        
        # Decisions (never cut)
        decisions = []
        for node_id, data in graph.nx_graph.nodes(data=True):
            nd = data["data"]
            if nd.is_decision and nd.darkness > 0.5:
                decisions.append(f"- {nd.id}")
                
        decisions_section = ""
        if decisions:
            decisions_section = "[ESTABLISHED DECISIONS - ALWAYS HONOUR THESE]\n" + "\n".join(decisions) + "\n\n"
            
        # Active context & Priority
        active_concepts = router.get_active_concepts(graph, top_k=config.router.top_k_concepts)
        priority_section = f"Priority concepts: {', '.join(active_concepts)}\n"
        graph_summary = graph.get_context_summary(active_concepts)
        
        active_section = f"[ACTIVE CONTEXT]\n{priority_section}\n{graph_summary}\n\n"
        
        # Gaps / Contradictions
        gaps = retrieved_data.get("gaps", [])
        contra = retrieved_data.get("contradictions", [])
        
        gaps_section = ""
        if gaps:
            gaps_section = "[GAPS DETECTED]\n" + "\n".join([f"- {g}" for g in gaps]) + "\n\n"
            
        contra_section = ""
        if contra:
            contra_section = "[CONTRADICTIONS DETECTED]\n" + "\n".join([f"- {c}" for c in contra]) + "\n\n"
            
        # History — use clean history list (user/assistant pairs), NOT graph.episodes
        history_section = "[RECENT TURNS]\n"
        for h in history[-config.brain.max_history_turns:]:
            history_section += f"User: {h['input']}\nAssistant: {h['response']}\n"
        history_section += "\n"
        
        current_request = f"[CURRENT REQUEST]\n{user_input}\n"
        
        # Assemble (System -> Decisions -> Request -> Graph -> Gaps -> History)
        prompt = self.system_prompt + "\n" + session_context + decisions_section + current_request
        base_tokens = self._estimate_tokens(prompt)
        
        # Enforce budget
        available = config.prompt.max_tokens - base_tokens
        
        if available > 0:
            active_tokens = self._estimate_tokens(active_section)
            if active_tokens < available:
                prompt += active_section
                available -= active_tokens
            else:
                prompt += active_section[:available * 4]
                available = 0
                
        if available > 0:
            gaps_contra = gaps_section + contra_section
            gc_tokens = self._estimate_tokens(gaps_contra)
            if gc_tokens < available:
                prompt += gaps_contra
                available -= gc_tokens
            else:
                prompt += gaps_contra[:available * 4]
                available = 0
                
        if available > 0:
            hist_tokens = self._estimate_tokens(history_section)
            if hist_tokens < available:
                prompt += history_section
                available -= hist_tokens
            else:
                prompt += history_section[:available * 4]
                
        return prompt
