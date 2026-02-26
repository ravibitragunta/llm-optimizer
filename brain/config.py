import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    model_path: str = os.getenv("MODEL_PATH", "data/models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf")
    n_ctx: int = int(os.getenv("MODEL_N_CTX", "8192"))
    n_threads: int = int(os.getenv("MODEL_N_THREADS", "4"))
    # n_batch: prompt processing parallelism — larger = faster KV fill, more RAM
    n_batch: int = int(os.getenv("MODEL_N_BATCH", "512"))
    # use_mlock: pin model weights in RAM, prevents OS paging mid-inference
    use_mlock: bool = os.getenv("MODEL_USE_MLOCK", "true").lower() == "true"
    # use_mmap: memory-map the model file for fast loading
    use_mmap: bool = os.getenv("MODEL_USE_MMAP", "true").lower() == "true"
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "1024"))
    repeat_penalty: float = float(os.getenv("MODEL_REPEAT_PENALTY", "1.1"))

@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    ttl_session: int = int(os.getenv("REDIS_TTL_SESSION", "86400"))
    ttl_user: int = int(os.getenv("REDIS_TTL_USER", "2592000"))
    ttl_pattern: int = int(os.getenv("REDIS_TTL_PATTERN", "2592000"))

@dataclass
class GraphConfig:
    max_nodes: int = int(os.getenv("GRAPH_MAX_NODES", "500"))
    max_edges: int = int(os.getenv("GRAPH_MAX_EDGES", "2000"))
    relevance_hops: int = int(os.getenv("GRAPH_RELEVANCE_HOPS", "2"))
    max_context_tokens: int = int(os.getenv("GRAPH_MAX_CONTEXT_TOKENS", "300"))
    darkness_threshold: float = float(os.getenv("GRAPH_DARKNESS_THRESHOLD", "0.1"))
    darkness_decay: float = float(os.getenv("GRAPH_DARKNESS_DECAY", "0.95"))
    darkness_increment: float = float(os.getenv("GRAPH_DARKNESS_INCREMENT", "0.3"))
    edge_increment: float = float(os.getenv("GRAPH_EDGE_INCREMENT", "0.2"))
    max_summary_relations: int = int(os.getenv("GRAPH_MAX_SUMMARY_RELATIONS", "10"))

@dataclass
class RouterConfig:
    state_dim: int = int(os.getenv("ROUTER_STATE_DIM", "256"))
    decay_rate: float = float(os.getenv("ROUTER_DECAY_RATE", "0.9"))
    update_rate: float = float(os.getenv("ROUTER_UPDATE_RATE", "0.1"))
    top_k_concepts: int = int(os.getenv("ROUTER_TOP_K_CONCEPTS", "5"))
    confidence_threshold: float = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.3"))

@dataclass
class RetrieverConfig:
    top_k_patterns: int = int(os.getenv("RETRIEVER_TOP_K_PATTERNS", "3"))
    similarity_threshold: float = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", "0.7"))
    min_pattern_frequency: int = int(os.getenv("RETRIEVER_MIN_PATTERN_FREQUENCY", "2"))

@dataclass
class PromptConfig:
    max_tokens: int = int(os.getenv("PROMPT_MAX_TOKENS", "2800"))
    graph_tokens: int = int(os.getenv("PROMPT_GRAPH_TOKENS", "300"))
    history_tokens: int = int(os.getenv("PROMPT_HISTORY_TOKENS", "400"))
    routing_tokens: int = int(os.getenv("PROMPT_ROUTING_TOKENS", "100"))

@dataclass
class BrainConfig:
    max_history_turns: int = int(os.getenv("BRAIN_MAX_HISTORY_TURNS", "3"))
    domain_detection_threshold: int = int(os.getenv("BRAIN_DOMAIN_DETECTION_THRESHOLD", "2"))

class AppConfig:
    def __init__(self):
        self.model = ModelConfig()
        self.redis = RedisConfig()
        self.graph = GraphConfig()
        self.router = RouterConfig()
        self.retriever = RetrieverConfig()
        self.prompt = PromptConfig()
        self.brain = BrainConfig()

config = AppConfig()
