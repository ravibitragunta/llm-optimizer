from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Entity:
    text: str
    entity_type: str   # 'class', 'function', 'concept', 'numeric_fact', 'decision'
    confidence: float
    verbatim_value: Optional[str] = None

@dataclass
class StructuredSummary:
    decisions: list = field(default_factory=list)          # list[str]
    active_concepts: list = field(default_factory=list)    # list[str]
    contradictions: list = field(default_factory=list)     # list[tuple[str,str]]
    verbatim_facts: dict = field(default_factory=dict)     # dict[str, str]
    gaps: list = field(default_factory=list)               # list[str]
    current_intent: str = ""

class ConversationPlugin(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def file_extensions(self) -> list:
        return ['*']

    @abstractmethod
    def extract_entities(self, text: str) -> list: ...

    @abstractmethod
    def detect_decisions(self, text: str) -> list: ...

    def detect_contradictions(self, existing: list, new_text: str) -> list:
        return []

    def get_seed_domains(self) -> list:
        return []

    def get_system_prompt(self) -> str:
        return "You are a helpful assistant."

    def compress_error(self, error_text: str) -> str:
        return '\n'.join(error_text.strip().split('\n')[:3])
