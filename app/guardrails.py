import logging
from typing import Optional, Tuple
from app.schemas import RetrievalResponse, GenerationResult, RefusalReason

logger = logging.getLogger(__name__)

# Default empirical threshold for bge-small cosine similarity
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

UNSAFE_KEYWORDS = [
    "हथियार", "बम", "हिंसा", "नफरत", "hack", "exploit", "kill", "attack"
]

class GuardrailGate:
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def evaluate_pre_generation(self, query: str, retrieval_resp: RetrievalResponse) -> Tuple[bool, Optional[RefusalReason]]:
        """Pre-generation hard gate checking query safety & appropriateness."""
        # 1. Check for unsafe / inappropriate query content
        query_lower = query.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in query_lower:
                reason = RefusalReason(
                    code="HARMFUL_QUERY",
                    message="क्षमा करें, आपका प्रश्न सुरक्षा नीतियों के अनुकूल नहीं है।",
                    score=0.0
                )
                logger.warning(f"[Guardrail Veto] Pre-generation unsafe query detected: '{query}'")
                return False, reason

        return True, None

    def evaluate_post_generation(self, query: str, gen_result: GenerationResult) -> Tuple[bool, Optional[RefusalReason]]:
        """Post-generation hard gate checking groundedness and output safety."""
        if not gen_result.grounded:
            reason = RefusalReason(
                code="UNGROUNDED_OUTPUT",
                message="उत्पन्न उत्तर संदर्भ में समर्थित नहीं है।",
                score=None
            )
            logger.warning(f"[Guardrail Veto] Post-generation output marked ungrounded for query: '{query}'")
            return False, reason

        answer_lower = gen_result.answer.lower()
        if "उपलब्ध नहीं है" in answer_lower:
            reason = RefusalReason(
                code="NOT_FOUND_IN_CONTEXT",
                message="दिए गए संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है।",
                score=None
            )
            logger.warning(f"[Guardrail Veto] Post-generation 'Not found in context' response for query: '{query}'")
            return False, reason

        return True, None
