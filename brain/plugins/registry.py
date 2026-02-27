import os
import redis
import json
from pathlib import Path
from brain.plugins.base import ConversationPlugin

class PluginRegistry:
    def __init__(self):
        self._plugins: dict = {}
        self._register_defaults()

    def _register_defaults(self):
        try:
            from brain.plugins.java_spring import JavaSpringPlugin
            from brain.plugins.general import GeneralPlugin
            from brain.plugins.sap import SAPPlugin
            for p in [JavaSpringPlugin(), SAPPlugin(), GeneralPlugin()]:
                self.register(p)
        except Exception as e:
            # Fallback to general only
            from brain.plugins.general import GeneralPlugin
            self.register(GeneralPlugin())

    def register(self, plugin: ConversationPlugin):
        self._plugins[plugin.name] = plugin

    def detect(self, text: str = '', file_path: str = '', explicit: str = '') -> ConversationPlugin:
        try:
            if explicit and explicit in self._plugins:
                return self._plugins[explicit]

            if file_path:
                ext = Path(file_path).suffix
                for plugin in self._plugins.values():
                    if ext in plugin.file_extensions and plugin.name != 'general':
                        return plugin

            if text:
                scores = {
                    name: self._score_text(text, plugin)
                    for name, plugin in self._plugins.items()
                    if name != 'general'
                }
                if scores:
                    best = max(scores, key=scores.get)
                    if scores[best] > 0.25:
                        return self._plugins[best]

            return self._plugins.get('general', list(self._plugins.values())[0])
        except Exception:
            return self._plugins.get('general', list(self._plugins.values())[0])

    def _score_text(self, text: str, plugin: ConversationPlugin) -> float:
        total, matched = 0, 0
        for domain in plugin.get_seed_domains():
            try:
                r = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
                raw = r.get(f"brain:lfm:domain:{domain}:knowledge")
                if raw:
                    seed = json.loads(raw)
                    kws = seed.get('trigger_concepts', [])
                    total += len(kws)
                    matched += sum(1 for kw in kws if kw.lower() in text.lower())
            except Exception:
                pass
        return matched / total if total > 0 else 0.0
