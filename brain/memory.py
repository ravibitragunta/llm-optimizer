import redis
import json
from typing import Dict, Any, List, Tuple
from brain.config import config
from brain.graph import ContextGraph
from brain.router.decay_router import DecayRouter

class Memory:
    def __init__(self):
        self.r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True
        )
        
    def save_session(self, session_id: str, graph: ContextGraph, router: DecayRouter, history: List[Dict], metrics: List[Dict]):
        pipe = self.r.pipeline()
        pipe.set(f"brain:session:{session_id}:graph", graph.serialize())
        pipe.set(f"brain:session:{session_id}:router", router.serialize())
        pipe.set(f"brain:session:{session_id}:history", json.dumps(history))
        pipe.set(f"brain:session:{session_id}:metrics", json.dumps(metrics))
        
        # Expire
        pipe.expire(f"brain:session:{session_id}:graph", config.redis.ttl_session)
        pipe.expire(f"brain:session:{session_id}:router", config.redis.ttl_session)
        pipe.expire(f"brain:session:{session_id}:history", config.redis.ttl_session)
        pipe.expire(f"brain:session:{session_id}:metrics", config.redis.ttl_session)
        
        pipe.execute()
        
    def load_session(self, session_id: str) -> Tuple[ContextGraph, DecayRouter, List[Dict], List[Dict]]:
        pipe = self.r.pipeline()
        pipe.get(f"brain:session:{session_id}:graph")
        pipe.get(f"brain:session:{session_id}:router")
        pipe.get(f"brain:session:{session_id}:history")
        pipe.get(f"brain:session:{session_id}:metrics")
        
        g_data, r_data, h_data, m_data = pipe.execute()
        
        graph = ContextGraph.deserialize(g_data) if g_data else ContextGraph()
        router = DecayRouter.deserialize(r_data) if r_data else DecayRouter()
        history = json.loads(h_data) if h_data else []
        metrics = json.loads(m_data) if m_data else []
        
        return graph, router, history, metrics
