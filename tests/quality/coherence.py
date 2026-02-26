import spacy

# Load the same spaCy model used by the graph
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")


class CoherenceChecker:
    """
    Measures how well the model's response reflects the active graph concepts.

    v2: Uses lemmatisation + substring matching instead of exact string matching.
    This prevents false negatives where e.g. node="constructor injection" but
    response says "constructor-injected" or "injected via constructor".
    """

    def _lemmatise(self, text: str) -> set:
        """Tokenise and lemmatise text, return a set of lowercase lemmas."""
        doc = nlp(text.lower())
        return {token.lemma_ for token in doc if not token.is_punct and not token.is_space}

    def _concept_present(self, concept_normalized: str, response_lemmas: set, response_lower: str) -> bool:
        """
        Return True if the concept is detectable in the response via any of:
          1. Exact substring match (fast path for class names / annotations)
          2. All content words of the concept appear as lemmas in the response
        """
        # Fast path: exact substring (catches CamelCase class names, annotations)
        if concept_normalized in response_lower:
            return True

        # Lemma path: split the concept into words, check each is present as a lemma
        concept_words = concept_normalized.split()
        if len(concept_words) > 1:
            # Multi-word concept — all significant words must appear
            significant = [w for w in concept_words if len(w) > 2]
            return all(w in response_lemmas for w in significant)

        # Single word concept — check lemma set
        return concept_normalized in response_lemmas

    def check(self, response: str, graph, turn: int) -> float:
        """
        Returns a coherence score in [0.0, 1.0]:
          1.0 = all active graph concepts (darkness > 0.5) are present in response
          0.0 = none of them are
        """
        response_lower = response.lower()
        response_lemmas = self._lemmatise(response)

        expected = 0
        hits = 0

        for node_id, data in graph.nx_graph.nodes(data=True):
            nd = data["data"]
            if nd.darkness > 0.5:
                expected += 1
                if self._concept_present(nd.normalized, response_lemmas, response_lower):
                    hits += 1

        if expected == 0:
            return 1.0

        return hits / expected
