import re
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

from brain.plugins.base import ConversationPlugin, Entity

class GeneralPlugin(ConversationPlugin):

    @property
    def name(self) -> str:
        return "general"

    def extract_entities(self, text: str) -> list:
        entities = []
        seen = set()

        # Named entities via spaCy
        if _nlp:
            doc = _nlp(text)
            for ent in doc.ents:
                if ent.label_ in ('ORG','PRODUCT','PERSON','GPE','EVENT','LAW','MONEY','DATE','PERCENT'):
                    verbatim = ent.text if ent.label_ in ('MONEY','DATE','PERCENT') else None
                    if ent.text not in seen:
                        entities.append(Entity(ent.text, ent.label_.lower(), 0.9, verbatim))
                        seen.add(ent.text)

            # Multi-word noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2 and chunk.text not in seen:
                    entities.append(Entity(chunk.text, 'concept', 0.7))
                    seen.add(chunk.text)

        # Numeric facts — always capture verbatim
        for match in re.finditer(
            r'\d+[\.\d]*\s*(?:%|USD|EUR|GBP|days?|months?|years?|hours?|\$|€|£)',
            text
        ):
            if match.group() not in seen:
                entities.append(Entity(match.group(), 'numeric_fact', 0.95, match.group()))
                seen.add(match.group())

        # CamelCase terms (class names in any OOP language)
        for match in re.finditer(r'\b[A-Z][a-zA-Z]{2,}(?:[A-Z][a-zA-Z]+)+\b', text):
            if match.group() not in seen:
                entities.append(Entity(match.group(), 'term', 0.6))
                seen.add(match.group())

        return entities

    def detect_decisions(self, text: str) -> list:
        if not _nlp:
            return []
        doc = _nlp(text)
        decisions = []
        decision_verbs = {'use', 'require', 'must', 'shall', 'will', 'always', 'never'}
        for sent in doc.sents:
            if any(tok.lemma_.lower() in decision_verbs for tok in sent):
                decisions.append(sent.text.strip())
        return decisions[:3]

    def get_system_prompt(self) -> str:
        return "You are a knowledgeable assistant. Be precise and factual."
