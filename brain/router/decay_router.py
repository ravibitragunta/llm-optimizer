import numpy as np
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass

from brain.config import config
from brain.graph import ContextGraph

@dataclass
class RouterData:
    state: List[float]
    turn_count: int
    current_domain: str

class DecayRouter:
    def __init__(self):
        self.state = np.zeros(config.router.state_dim)
        self.turn_count = 0
        self.current_domain = "unknown"
        
        self.DOMAIN_MAP = {
            "spring_jdbc": (128, 136),
            "spring_boot": (136, 144),
            "spring_security": (144, 152),
            "spring_async": (152, 160),
            "java_general": (160, 168)
        }
        
    def _hash_to_dim(self, text: str, offset: int, space_size: int = 64) -> int:
        h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        return (h % space_size) + offset

    def encode_graph_to_signal(self, graph: ContextGraph, active_nodes: List[str]) -> np.ndarray:
        signal = np.zeros(config.router.state_dim)
        
        # Entities (0-63)
        for node_id in active_nodes:
            if graph.nx_graph.has_node(node_id):
                node = graph.nx_graph.nodes[node_id]["data"]
                dim = self._hash_to_dim(node.normalized, offset=0)
                signal[dim] += node.darkness
                
        # Strong Relationships (64-127)
        for u, v, ed_dict in graph.nx_graph.edges(data=True):
            ed = ed_dict["data"]
            if ed.weight > 0.3:
                dim = self._hash_to_dim(ed.type + ed.target, offset=64)
                signal[dim] += ed.weight
                
        # Domains (128-191)
        if self.current_domain in self.DOMAIN_MAP:
            start, end = self.DOMAIN_MAP[self.current_domain]
            # one hot style block assignment
            signal[start:end] = 1.0
            
        # Normalize
        norm = np.linalg.norm(signal)
        if norm > 0:
            signal = signal / norm
            
        return signal

    def update(self, graph: ContextGraph, active_nodes: List[str]):
        signal = self.encode_graph_to_signal(graph, active_nodes)
        self.state = self.state * config.router.decay_rate + signal * config.router.update_rate
        self.turn_count += 1
        
        # update domain internally after state update
        self.current_domain = self.get_domain()

    def encode_single_node(self, node) -> np.ndarray:
        signal = np.zeros(config.router.state_dim)
        dim = self._hash_to_dim(node.normalized, offset=0)
        signal[dim] = 1.0
        return signal

    def score_nodes(self, graph: ContextGraph) -> Dict[str, float]:
        scores = {}
        for node_id, data in graph.nx_graph.nodes(data=True):
            node = data["data"]
            node_signal = self.encode_single_node(node)
            score = np.dot(self.state, node_signal)
            scores[node.id] = score
        return scores

    def get_active_concepts(self, graph: ContextGraph, top_k: int = 5) -> List[str]:
        scores = self.score_nodes(graph)
        sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [node_id for node_id, _ in sorted_nodes[:top_k]]

    def get_confidence(self) -> float:
        # Assuming maximum steady state norm is ~ 1.0 depending on updates
        n = np.linalg.norm(self.state)
        return min(1.0, float(n))

    def get_domain(self) -> str:
        domain_scores = {}
        for domain, (start, end) in self.DOMAIN_MAP.items():
            domain_scores[domain] = np.mean(self.state[start:end])
            
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        if best_domain[1] > 0.05: # threshold logic
            return best_domain[0]
        return "unknown"

    def serialize(self) -> str:
        data = {
            "state": self.state.tolist(),
            "turn_count": self.turn_count,
            "current_domain": self.current_domain
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data_str: str) -> 'DecayRouter':
        router = cls()
        data = json.loads(data_str)
        router.state = np.array(data["state"])
        router.turn_count = data["turn_count"]
        router.current_domain = data["current_domain"]
        return router
