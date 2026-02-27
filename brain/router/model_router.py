import os
from dataclasses import dataclass

@dataclass
class ModelDecision:
    model: str
    endpoint: str
    reason: str
    estimated_cost_usd: float

class ModelRouter:
    COSTS = {
        'local':  0.0,
        'haiku':  0.000_002,
        'sonnet': 0.000_010,
        'opus':   0.000_015,
        'gpt4o':  0.000_015,
        'grok':   0.000_008,
    }

    def route(self, session_state: dict, request: str, prompt_tokens: int) -> ModelDecision:
        has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY', ''))
        has_openai    = bool(os.getenv('OPENAI_API_KEY', ''))
        has_grok      = bool(os.getenv('XAI_API_KEY', ''))
        has_cloud     = has_anthropic or has_openai or has_grok

        if not has_cloud:
            return ModelDecision('local', 'local_gguf', 'no cloud keys configured', 0.0)

        complexity = self._score_complexity(session_state, request)

        if complexity < 0.40:
            return ModelDecision('local', 'local_gguf', f'complexity {complexity:.2f} < 0.40', 0.0)

        if complexity < 0.65:
            if has_anthropic:
                return ModelDecision('haiku', 'anthropic', f'complexity {complexity:.2f}', self.COSTS['haiku'] * prompt_tokens)
            if has_openai:
                return ModelDecision('gpt4o', 'openai', f'complexity {complexity:.2f}', self.COSTS['gpt4o'] * prompt_tokens)

        if complexity < 0.85:
            if has_anthropic:
                return ModelDecision('sonnet', 'anthropic', f'complexity {complexity:.2f}', self.COSTS['sonnet'] * prompt_tokens)
            if has_openai:
                return ModelDecision('gpt4o', 'openai', f'complexity {complexity:.2f}', self.COSTS['gpt4o'] * prompt_tokens)

        if has_anthropic:
            return ModelDecision('opus', 'anthropic', f'max complexity {complexity:.2f}', self.COSTS['opus'] * prompt_tokens)

        return ModelDecision('local', 'local_gguf', 'no suitable cloud provider', 0.0)

    def _score_complexity(self, state: dict, request: str) -> float:
        r = request.lower()
        signals = [
            state.get('graph_nodes', 0) > 40,
            state.get('turn', 0) > 12,
            state.get('contradictions_count', 0) > 0,
            state.get('gaps_count', 0) > 3,
            state.get('routing_confidence', 0) > 0.7,
            any(w in r for w in ['implement','architect','design','analyse','refactor','debug','explain why','compare']),
            '?' in request and any(w in r for w in ['why','how does','what is the difference']),
            not any(w in r for w in ['ok','yes','got it','continue','next','sure','thanks']),
        ]
        return sum(signals) / len(signals) if signals else 0.0
