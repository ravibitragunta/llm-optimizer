import os
from typing import Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

class CloudClient:
    def __init__(self):
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.openai_key    = os.getenv('OPENAI_API_KEY', '')
        self.grok_key      = os.getenv('XAI_API_KEY', '')

    def generate(self, prompt: str, model_decision, max_tokens: int = 1024) -> str:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Run: pip install httpx>=0.27.0")

        if model_decision.endpoint == 'local_gguf':
            raise ValueError("Use local Llama instance for local inference")

        if model_decision.endpoint == 'anthropic':
            return self._anthropic(prompt, model_decision.model, max_tokens)

        if model_decision.endpoint == 'openai':
            return self._openai(prompt, model_decision.model, max_tokens)

        raise ValueError(f"Unknown endpoint: {model_decision.endpoint}")

    def _split_prompt(self, prompt: str) -> tuple:
        if '<|im_start|>system' in prompt:
            parts = prompt.split('<|im_end|>\n')
            system = parts[0].replace('<|im_start|>system\n', '').strip()
            user_raw = parts[1] if len(parts) > 1 else ''
            user = user_raw.replace('<|im_start|>user\n', '').replace('<|im_end|>\n', '').strip()
            return system, user
        elif '<|start_header_id|>system<|end_header_id|>' in prompt:
            system_part = prompt.split('<|eot_id|>')[0]
            system = system_part.split('<|end_header_id|>\n')[1].strip()
            user = 'Continue the session based on the context above.'
            return system, user
        else:
            return "You are a helpful assistant.", prompt

    def _anthropic(self, prompt: str, model: str, max_tokens: int) -> str:
        model_ids = {
            'haiku':  'claude-haiku-4-5',
            'sonnet': 'claude-sonnet-4-5',
            'opus':   'claude-opus-4-5',
        }
        model_id = model_ids.get(model, 'claude-sonnet-4-5')
        system, user = self._split_prompt(prompt)

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()['content'][0]['text']

    def _openai(self, prompt: str, model: str, max_tokens: int) -> str:
        model_ids = {'gpt4o': 'gpt-4o', 'gpt4mini': 'gpt-4o-mini'}
        model_id = model_ids.get(model, 'gpt-4o')
        system, user = self._split_prompt(prompt)

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_key}"},
            json={
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
