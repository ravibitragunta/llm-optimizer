import redis
import json
import time
from typing import Set, List, Dict, Any, Optional

from brain.config import config
from brain.graph import ContextGraph
from brain.router.decay_router import DecayRouter

class Retriever:
    def __init__(self):
        self.r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True
        )
        
    def _jaccard(self, set_a: Set[int], set_b: Set[int]) -> float:
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def get_top_active_dims(self, state: list, top_n: int = 20) -> Set[int]:
        indexed = [(i, val) for i, val in enumerate(state)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return set([i for i, val in indexed[:top_n] if val > 0])

    def store_pattern(self, active_dims: Set[int], next_concepts: List[str], domain: str):
        if not active_dims:
            return
            
        sorted_dims = tuple(sorted(list(active_dims)))
        pattern_hash = str(hash(sorted_dims))
        key = f"brain:lfm:patterns:{pattern_hash}"
        
        existing = self.r.get(key)
        if existing:
            data = json.loads(existing)
            data["frequency"] += 1
            data["next_concepts"] = list(set(data["next_concepts"] + next_concepts))
            data["last_seen"] = int(time.time())
            self.r.setex(key, config.redis.ttl_pattern, json.dumps(data))
        else:
            data = {
                "active_dims": list(active_dims),
                "next_concepts": next_concepts,
                "domain": domain,
                "frequency": 1,
                "last_seen": int(time.time())
            }
            self.r.setex(key, config.redis.ttl_pattern, json.dumps(data))

    def retrieve_similar_patterns(self, active_dims: Set[int]) -> List[Dict]:
        results = []
        if not active_dims:
            return results
            
        keys = self.r.scan_iter("brain:lfm:patterns:*")
        for key in keys:
            data_str = self.r.get(key)
            if not data_str: continue
            data = json.loads(data_str)
            
            if data.get("frequency", 0) < config.retriever.min_pattern_frequency:
                continue
                
            pattern_dims = set(data["active_dims"])
            sim = self._jaccard(active_dims, pattern_dims)
            if sim > config.retriever.similarity_threshold:
                results.append({
                    "similarity": sim,
                    "next_concepts": data["next_concepts"],
                    "domain": data["domain"]
                })
                
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:config.retriever.top_k_patterns]

    def retrieve_domain_knowledge(self, domain: str) -> Optional[Dict]:
        key = f"brain:lfm:domain:{domain}:knowledge"
        data = self.r.get(key)
        if data:
            return json.loads(data)
        return None
        
    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def detect_gaps(self, graph: ContextGraph, domain_knowledge: Dict) -> List[str]:
        if not domain_knowledge: return []
        
        active_concepts_normalized = set([self._normalize(n) for n in graph.nx_graph.nodes])
        gaps_found = []
        
        for gap in domain_knowledge.get("common_gaps", []):
            if self._normalize(gap) not in active_concepts_normalized:
                gaps_found.append(gap)
                
        return gaps_found

    def detect_contradictions(self, graph: ContextGraph, domain_knowledge: Dict) -> List[str]:
        if not domain_knowledge: return []
        
        active_concepts = set([self._normalize(n) for n in graph.nx_graph.nodes])
        contradictions = []
        for pair in domain_knowledge.get("common_contradictions", []):
            if len(pair) == 2:
                if self._normalize(pair[0]) in active_concepts and self._normalize(pair[1]) in active_concepts:
                    contradictions.append(f"Contradiction detected: {pair[0]} vs {pair[1]}")
        
        # also check graph explicit contradict edges
        for u, v, data in graph.nx_graph.edges(data=True):
            if data["data"].type == "contradicts":
                contradictions.append(f"Explicit contradiction: {u} contradicts {v}")
                
        return contradictions

    def retrieve_all(self, graph: ContextGraph, router: DecayRouter, user_id: str, session_id: str) -> Dict[str, Any]:
        active_dims = self.get_top_active_dims(router.state.tolist())
        
        patterns = self.retrieve_similar_patterns(active_dims)
        domain_knowledge = self.retrieve_domain_knowledge(router.current_domain)
        
        gaps = self.detect_gaps(graph, domain_knowledge) if domain_knowledge else []
        contradictions = self.detect_contradictions(graph, domain_knowledge) if domain_knowledge else []
        
        return {
            "patterns": patterns,
            "domain_knowledge": domain_knowledge,
            "gaps": gaps,
            "contradictions": contradictions
        }
