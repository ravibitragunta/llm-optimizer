import re
import spacy
import networkx as nx
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

from brain.config import config

# Load spacy model globally to avoid loading it per turn
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not downloaded. Run: python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

# Define regex patterns
SPRING_CLASSES_REGEX = re.compile(r'\b[A-Z][a-zA-Z]+(?:Template|Service|Repository|Controller|Manager|Factory|Handler|Mapper|Config|Configuration|Exception|Filter|Interceptor|Advisor|Aspect|Bean|Component|Entity)\b')

SPRING_ANNOTATIONS_REGEX = re.compile(r'@(?:Repository|Service|Controller|RestController|Component|Configuration|Bean|Autowired|Transactional|RequestMapping|GetMapping|PostMapping|PutMapping|DeleteMapping|PathVariable|RequestBody|ResponseBody|SpringBootApplication|EnableAutoConfiguration|Async|Scheduled|EventListener|Valid|NotNull|NotEmpty|Size|Email|Min|Max|SpringBootTest|JdbcTest|WebMvcTest|MockBean|MockMvc)\b')

JDBC_SPECIFIC = ["NamedParameterJdbcTemplate", "JdbcTemplate", "SqlParameterSource", "MapSqlParameterSource", "BeanPropertySqlParameterSource", "RowMapper", "BeanPropertyRowMapper", "ResultSet", "PreparedStatement"]

SECURITY_SPECIFIC = ["SecurityFilterChain", "UserDetailsService", "JwtAuthenticationFilter", "AuthenticationManager", "PasswordEncoder", "BCryptPasswordEncoder"]

# Exception handling, REST responses, validation — appear in turns 6-10
SPRING_EXCEPTION_PATTERNS = [
    "GlobalExceptionHandler", "ControllerAdvice", "ExceptionHandler",
    "ResponseEntity", "HttpStatus", "ResponseStatus",
    "RuntimeException", "IllegalArgumentException", "IllegalStateException",
    "MethodArgumentNotValidException", "BindingResult", "FieldError",
    "@Valid", "@NotNull", "@NotBlank", "@NotEmpty", "@Size", "@Email",
    "@Min", "@Max", "@Pattern",
    "ProblemDetail", "ErrorResponse",
]

@dataclass
class NodeData:
    id: str
    normalized: str
    darkness: float = 0.0
    last_active_turn: int = 0
    activation_count: int = 0
    is_decision: bool = False
    
    def to_dict(self):
        return {
            "id": self.id,
            "normalized": self.normalized,
            "darkness": self.darkness,
            "last_active_turn": self.last_active_turn,
            "activation_count": self.activation_count,
            "is_decision": self.is_decision
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class EdgeData:
    source: str
    target: str
    type: str
    weight: float = 0.0
    last_active_turn: int = 0
    
    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "last_active_turn": self.last_active_turn
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class Episode:
    text: str
    turn_number: int
    active_nodes: List[str]

class ContextGraph:
    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self.episodes: List[Episode] = []
        self._last_active_edges: List[EdgeData] = []
        
    def _normalize(self, text: str) -> str:
        return text.strip().lower()
        
    def extract_entities(self, text: str) -> List[str]:
        entities = set()
        
        # Layer 1 - Regex and keywords
        entities.update(SPRING_CLASSES_REGEX.findall(text))
        entities.update(SPRING_ANNOTATIONS_REGEX.findall(text))
        
        for kw in JDBC_SPECIFIC + SECURITY_SPECIFIC + SPRING_EXCEPTION_PATTERNS:
            if kw in text:
                entities.add(kw)
                
        # Layer 2 & 3 - spaCy
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"] and len(ent.text) > 3:
                entities.add(ent.text)
                
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip("\n\t \"'")
            if 2 <= len(chunk_text) <= 50 and any(c.isalpha() for c in chunk_text):
                entities.add(chunk_text)
                
        # Deduplicate by normalized text
        unique_entities = {}
        for ent in entities:
            norm = self._normalize(ent)
            if norm not in unique_entities or len(ent) > len(unique_entities[norm]):
                unique_entities[norm] = ent
                
        return list(unique_entities.values())[:20]

    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        # A simple proximity/heuristic based extraction
        relationships = []
        doc = nlp(text.lower())
        
        # We need entities extracted from the text in their exact case for accurate linking
        entities = self.extract_entities(text)
        ent_norms = {self._normalize(e): e for e in entities}
        
        # Simplified pattern matching logic
        # In a real NLP setting we would parse the dependency tree. For MVP:
        
        # Find decision pattern
        decision_patterns = ["we decided", "let's use", "we'll use", "going with"]
        for dp in decision_patterns:
            if dp in text.lower():
                # Extract what follows
                idx = text.lower().find(dp) + len(dp)
                following_text = text[idx:idx+50].strip()
                if following_text:
                    # Treat the first entity found in following text as a decision
                    following_ents = [e for e in entities if self._normalize(e) in following_text.lower()]
                    for df in following_ents:
                        relationships.append((df, "decision", ""))
        
        # Negation / contradiction
        negation_patterns = ["don't use", "avoid", "instead of", "not use"]
        for np in negation_patterns:
            if np in text.lower():
                idx = text.lower().find(np) + len(np)
                following_text = text[idx:idx+50].strip()
                following_ents = [e for e in entities if self._normalize(e) in following_text.lower()]
                for df in following_ents:
                    relationships.append((df, "contradicts", "CURRENT_WIP"))
        
        # Confirmation
        confirmation_patterns = ["yes", "correct", "exactly", "right", "good", "perfect"]
        if any(cpf in text.lower().split()[:5] for cpf in confirmation_patterns): # Only check start of text
             relationships.append(("CONFIRMATION", "strengthen", ""))
             
        # "uses", "extends", "implements"
        patterns = {
            "uses": ["uses", "using", "with", "via"],
            "extends": ["extends", "extends from"],
            "implements": ["implements", "implementing"],
            "injection_style": ["inject", "injected via", "constructor"],
            "annotated_with": ["annotated with", "annotate"],
            "returns": ["returns", "return type"],
            "throws": ["throws", "throw"]
        }
        
        for rel_type, triggers in patterns.items():
            for t in triggers:
               if t in text.lower():
                   relationships.append(("SYSTEM", rel_type, text.lower())) # MVP simplification

        return relationships

    def find_or_create(self, entity: str) -> NodeData:
        norm = self._normalize(entity)
        for node_id, data in self.nx_graph.nodes(data=True):
            nd = data["data"]
            if nd.normalized == norm:
                return nd
                
        # Create
        nd = NodeData(id=entity, normalized=norm)
        self.nx_graph.add_node(entity, data=nd)
        return nd
        
    def find_or_create_edge(self, source: str, target: str, rel_type: str) -> EdgeData:
        if self.nx_graph.has_edge(source, target):
            edge_data = self.nx_graph[source][target]["data"]
            if edge_data.type == rel_type:
                return edge_data
        
        # Create
        ed = EdgeData(source=source, target=target, type=rel_type)
        self.nx_graph.add_edge(source, target, data=ed)
        return ed

    def ingest(self, text: str, turn_number: int) -> List[str]:
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text)
        
        active_nodes = []
        new_active_edges = []
        
        for entity in entities:
            node = self.find_or_create(entity)
            node.darkness += config.graph.darkness_increment
            node.last_active_turn = turn_number
            node.activation_count += 1
            active_nodes.append(node.id)
            
        for (src, rel_type, tgt) in relationships:
            if rel_type == "decision":
                node = self.find_or_create(src)
                node.is_decision = True
                node.darkness = max(node.darkness, 1.5)
            elif rel_type == "strengthen":
                for ed in self._last_active_edges[:3]:
                    ed.weight += config.graph.edge_increment
            elif tgt:  # For standard relational edges
                # Find best target node from active
                possible_targets = [n for n in active_nodes if self._normalize(n) in tgt]
                actual_target = possible_targets[0] if possible_targets else src # Fallback
                if src != actual_target:
                    self.find_or_create(src)
                    self.find_or_create(actual_target)
                    edge = self.find_or_create_edge(src, actual_target, rel_type)
                    edge.weight += config.graph.edge_increment
                    edge.last_active_turn = turn_number
                    new_active_edges.append(edge)
                
        if new_active_edges:
           self._last_active_edges = new_active_edges
           
        self.episodes.append(Episode(text, turn_number, active_nodes))
        return active_nodes

    def decay(self, turn_number: int):
        nodes_to_remove = []
        edges_to_remove = []
        
        for node_id, data in self.nx_graph.nodes(data=True):
            nd = data["data"]
            if nd.is_decision:
                nd.darkness = max(0.5, nd.darkness * config.graph.darkness_decay)
            elif nd.activation_count > 5:
                 nd.darkness = max(0.3, nd.darkness * config.graph.darkness_decay)
            else:
                 nd.darkness *= config.graph.darkness_decay
                 if nd.darkness < config.graph.darkness_threshold:
                     nodes_to_remove.append(node_id)
                     
        for u, v, data in self.nx_graph.edges(data=True):
             ed = data["data"]
             ed.weight *= config.graph.darkness_decay
             if ed.weight < 0.05:
                 edges_to_remove.append((u, v))
                 
        for node_id in nodes_to_remove:
            self.nx_graph.remove_node(node_id)
            
        for u, v in edges_to_remove:
            if self.nx_graph.has_edge(u, v):
               self.nx_graph.remove_edge(u, v)

    def get_context_summary(self, active_nodes: List[str] = None) -> str:
        # Section 1 - Decisions
        decisions = []
        for node_id, data in self.nx_graph.nodes(data=True):
             nd = data["data"]
             if nd.is_decision and nd.darkness > 0.5:
                 decisions.append(f"DECISION: {nd.id}")
        
        decisions = decisions[:5]
        
        # Section 2 - Active concepts
        all_nodes = [data["data"] for _, data in self.nx_graph.nodes(data=True)]
        all_nodes.sort(key=lambda x: x.darkness, reverse=True)
        concepts = [f"{n.id} (used {n.activation_count}x)" for n in all_nodes[:10]]
        
        # Section 3 - Key relationships
        all_edges = [data["data"] for u, v, data in self.nx_graph.edges(data=True)]
        all_edges.sort(key=lambda x: x.weight, reverse=True)
        relations = [f"{e.source} --[{e.type}]--> {e.target}" for e in all_edges[:config.graph.max_summary_relations]]
        
        # Section 4 - Recent episodes
        recent_episodes = [ep.text[:100] + "..." if len(ep.text) > 100 else ep.text for ep in self.episodes[-2:]]
        
        summary = ""
        if decisions:
            summary += "[ESTABLISHED DECISIONS]\n" + "\n".join(decisions) + "\n\n"
        if concepts:
            summary += "[ACTIVE CONCEPTS]\n" + "\n".join(concepts) + "\n\n"
        if relations:
             summary += "[RELATIONSHIPS]\n" + "\n".join(relations) + "\n\n"
        if recent_episodes:
             summary += "[RECENT PROGRESS]\n" + "\n".join(recent_episodes) + "\n\n"
             
        # Token estimation (chars / 4)
        if len(summary) / 4 > config.graph.max_context_tokens:
            summary = summary[:int(config.graph.max_context_tokens * 4)]
            
        return summary
        
    def serialize(self) -> str:
        data = {
           "nodes": [d["data"].to_dict() for _, d in self.nx_graph.nodes(data=True)],
           "edges": [d["data"].to_dict() for _, _, d in self.nx_graph.edges(data=True)]
        }
        return json.dumps(data)
        
    @classmethod
    def deserialize(cls, data_str: str) -> 'ContextGraph':
        cg = cls()
        data = json.loads(data_str)
        for nd_dict in data.get("nodes", []):
             nd = NodeData.from_dict(nd_dict)
             cg.nx_graph.add_node(nd.id, data=nd)
             
        for ed_dict in data.get("edges", []):
             ed = EdgeData.from_dict(ed_dict)
             if cg.nx_graph.has_node(ed.source) and cg.nx_graph.has_node(ed.target):
                 cg.nx_graph.add_edge(ed.source, ed.target, data=ed)
        return cg
