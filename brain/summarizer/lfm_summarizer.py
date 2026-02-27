import json
import os
from typing import Optional
from brain.plugins.base import StructuredSummary

LFM_EXTRACTION_PROMPT = """Extract structured information from this conversation.
Return ONLY valid JSON with these exact keys. No markdown, no preamble.

{
  "decisions": ["established architectural/factual decisions as precise strings"],
  "active_concepts": ["concepts currently being discussed"],
  "contradictions": [["concept_a", "concept_b"]],
  "verbatim_facts": {"label": "exact value — preserve all numbers, dates, monetary amounts verbatim"},
  "gaps": ["things implied but not yet addressed in the conversation"],
  "current_intent": "what the user is trying to accomplish right now in one sentence"
}

Rules:
- verbatim_facts MUST capture all numbers, percentages, dates, monetary values EXACTLY as stated
- decisions must be stated facts ("we will use X"), not questions or maybes
- Keep each string under 100 characters
- Return valid JSON ONLY — no preamble, no explanation"""

from brain.config import config

class LFMSummarizer:
    def __init__(self, model_path: Optional[str] = None):
        path = model_path or config.brain.lfm_model_path
        self._model = None

        if not path:
            print("[LFMSummarizer] Model path not configured.")
            return

        if not os.path.exists(path):
            print(f"[LFMSummarizer] Model not found at {path}, running without LFM.")
            return

        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=path,
                n_ctx=32768,
                n_threads=int(os.getenv('MODEL_N_THREADS', '4')),
                n_batch=2048,
                use_mlock=True,
                use_mmap=True,
                verbose=False,
            )
            print(f"[LFMSummarizer] Loaded LFM from {path}")
        except Exception as e:
            print(f"[LFMSummarizer] Failed to load: {e}. Running without LFM.")

    @property
    def available(self) -> bool:
        return self._model is not None

    def summarize(self, full_history: list) -> StructuredSummary:
        if not self.available:
            return StructuredSummary()

        conversation_text = self._format_history(full_history)
        prompt = f"{LFM_EXTRACTION_PROMPT}\n\nConversation:\n{conversation_text}\n\nJSON:"

        try:
            response = self._model(
                prompt,
                max_tokens=1200,
                temperature=0.05,
                stop=['\n\n\n'],
            )
            raw = response['choices'][0]['text'].strip()

            if raw.startswith('```'):
                raw = '\n'.join(raw.split('\n')[1:])
                if raw.endswith('```'):
                    raw = raw[:-3]

            data = json.loads(raw)
            return StructuredSummary(
                decisions=data.get('decisions', []),
                active_concepts=data.get('active_concepts', []),
                contradictions=data.get('contradictions', []),
                verbatim_facts=data.get('verbatim_facts', {}),
                gaps=data.get('gaps', []),
                current_intent=data.get('current_intent', ''),
            )
        except Exception as e:
            print(f"[LFMSummarizer] Summarization failed: {e}")
            return StructuredSummary()

    def _format_history(self, history: list) -> str:
        lines = []
        for h in history:
            lines.append(f"Turn {h.get('turn', 0)} User: {h.get('input', '')}")
            resp = h.get('response', '')
            if len(resp) > 500:
                resp = resp[:500] + '...[truncated]'
            if resp:
                lines.append(f"Turn {h.get('turn', 0)} Assistant: {resp}")
        return '\n'.join(lines)
