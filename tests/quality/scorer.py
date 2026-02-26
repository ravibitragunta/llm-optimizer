from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class QualityScore:
    total: float
    compilability: float
    correctness: float
    consistency: float
    completeness: float
    convention: float

class SpringBootQualityScorer:
    def score(self, response_text: str, turn_config: Dict[str, Any], session_decisions: List[Any]) -> QualityScore:
        s1 = self.score_compilability(response_text)
        s2 = self.score_correctness(response_text, turn_config.get("required_keywords", {}))
        s3 = self.score_consistency(response_text, session_decisions)
        s4 = self.score_completeness(response_text)
        s5 = self.score_convention(response_text)
        
        return QualityScore(
            total=s1+s2+s3+s4+s5,
            compilability=s1,
            correctness=s2,
            consistency=s3,
            completeness=s4,
            convention=s5
        )

    def score_compilability(self, text: str) -> float:
        open_braces = text.count("{")
        close_braces = text.count("}")
        if abs(open_braces - close_braces) > 2: return 0.0
        if "public class" in text or "public interface" in text:
            if "}" not in text: return 0.0
        return 1.0

    def score_correctness(self, text: str, required_keywords: Dict[str, float]) -> float:
        score = 0.0
        for keyword, weight in required_keywords.items():
            if keyword.lower() in text.lower():
                score += weight
        return min(3.0, score)

    def score_consistency(self, text: str, session_decisions: List[Any]) -> float:
        score = 3.0
        for decision in session_decisions:
            d_type = getattr(decision, "type", str(decision).lower())
            if "constructor" in d_type and "injection" in d_type:
                if "@Autowired" in text and not "public" in text:
                    score -= 1.0
            if "jdbc" in d_type or "npjt" in d_type:
                if "JdbcTemplate" in text and "Named" not in text:
                    score -= 1.0
        return max(0.0, score)

    def score_completeness(self, text: str) -> float:
        score = 0.0
        has_error_handling = any(kw in text for kw in ["catch", "throw", "Optional", "exception", "handle", "if (", "null check"])
        has_edge_cases = any(kw in text for kw in ["empty", "null", "not found", "exists", "Optional.empty", "IllegalArgument"])
        
        if has_error_handling: score += 1.0
        if has_edge_cases: score += 1.0
        return score

    def score_convention(self, text: str) -> float:
        score = 0.0
        if "@Repository" in text: score += 1.0
        if "@Service" in text: score += 1.0
        if "@RestController" in text or "@Controller" in text:
            score = max(score, 1.0)
        if "final" in text and "private" in text: score += 0.5
        return min(2.0, score)
