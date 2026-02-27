import struct
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PromptFormat(Enum):
    CHATML   = "chatml"
    LLAMA2   = "llama2"
    LLAMA3   = "llama3"
    MISTRAL  = "mistral"
    PLAIN    = "plain"

@dataclass
class InferenceModelConfig:
    name: str
    context_window: int
    prompt_format: PromptFormat
    recommended_max_tokens: int
    recommended_temperature: float
    stop_tokens: list

    def format_prompt(self, system: str, context: str, request: str) -> str:
        if self.prompt_format == PromptFormat.CHATML:
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{context}\n\n{request}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif self.prompt_format == PromptFormat.LLAMA3:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{context}\n\n{request}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.prompt_format == PromptFormat.MISTRAL:
            return f"[INST] {system}\n\n{context}\n\n{request} [/INST]"
        else:
            return f"{system}\n\n{context}\n\nUser: {request}\nAssistant:"

MODEL_REGISTRY = {
    "qwen2.5-coder-3b":  InferenceModelConfig("qwen2.5-coder-3b",  32768, PromptFormat.CHATML,  512, 0.2, ["<|im_end|>"]),
    "qwen2.5-coder-7b":  InferenceModelConfig("qwen2.5-coder-7b",  32768, PromptFormat.CHATML,  512, 0.2, ["<|im_end|>"]),
    "qwen2.5-coder-32b": InferenceModelConfig("qwen2.5-coder-32b", 32768, PromptFormat.CHATML,  512, 0.2, ["<|im_end|>"]),
    "llama3":            InferenceModelConfig("llama3",             8192,  PromptFormat.LLAMA3,  512, 0.2, ["<|eot_id|>"]),
    "codellama":         InferenceModelConfig("codellama",          16384, PromptFormat.LLAMA2,  512, 0.2, ["</s>"]),
    "mistral":           InferenceModelConfig("mistral",            32768, PromptFormat.MISTRAL, 512, 0.2, ["</s>"]),
    "codestral":         InferenceModelConfig("codestral",          32768, PromptFormat.MISTRAL, 512, 0.2, ["</s>"]),
    "phi":               InferenceModelConfig("phi",                4096,  PromptFormat.CHATML,  512, 0.2, ["<|im_end|>"]),
    "gemma":             InferenceModelConfig("gemma",              8192,  PromptFormat.PLAIN,   512, 0.2, []),
    "lfm":               InferenceModelConfig("lfm",                32768, PromptFormat.CHATML, 1200, 0.05, ["<|im_end|>"]),
    "unknown":           InferenceModelConfig("unknown",            4096,  PromptFormat.PLAIN,   256, 0.3, []),
}

def detect_model_config(model_path: str) -> InferenceModelConfig:
    path_lower = model_path.lower()
    for key, cfg in MODEL_REGISTRY.items():
        if key in path_lower:
            return cfg
    try:
        result = _read_gguf_metadata(model_path)
        return result or MODEL_REGISTRY["unknown"]
    except Exception:
        return MODEL_REGISTRY["unknown"]

def _read_gguf_metadata(model_path: str) -> Optional[InferenceModelConfig]:
    try:
        with open(model_path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return None
            f.read(4)
            f.read(8)
            n_kv = struct.unpack('<Q', f.read(8))[0]
            meta = {}
            for _ in range(min(n_kv, 64)):
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8', errors='ignore')
                val_type = struct.unpack('<I', f.read(4))[0]
                if val_type == 8:
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    meta[key] = f.read(str_len).decode('utf-8', errors='ignore')
                else:
                    break
            template = meta.get('tokenizer.chat_template', '')
            fmt = (
                PromptFormat.CHATML   if 'im_start'      in template else
                PromptFormat.LLAMA3   if 'begin_of_text' in template else
                PromptFormat.MISTRAL  if '[INST]'         in template else
                PromptFormat.PLAIN
            )
            ctx = int(meta.get('general.context_length', 4096))
            return InferenceModelConfig(
                name=meta.get('general.name', 'unknown'),
                context_window=ctx,
                prompt_format=fmt,
                recommended_max_tokens=512,
                recommended_temperature=0.2,
                stop_tokens=[],
            )
    except Exception:
        return None
