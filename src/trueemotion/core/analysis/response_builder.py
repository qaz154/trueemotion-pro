"""
Response builder v1.18
======================
Steps 6-7 of the analysis flow:
  6. Generate empathy response (LLM or rule engine)
  7. Attach follow-up suggestion from context analysis
"""

import logging
from typing import Optional, Dict, List

from trueemotion.core.analysis.output import HumanResponse
from trueemotion.core.analysis.context import ContextualAnalyzer
from trueemotion.core.response.engine import HumanEmpathyEngine
from trueemotion.memory.repository import MemoryRepository

# LLM types (optional)
try:
    from trueemotion.core.llm import LLMResponseGenerator, FallbackManager
except ImportError:
    LLMResponseGenerator = None
    FallbackManager = None


class ResponseBuilder:
    """
    Builds the human-facing empathy response (steps 6-7).

    Initialised once per EmotionAnalyzer lifetime.
    """

    def __init__(
        self,
        rule_empathy: HumanEmpathyEngine,
        context_analyzer: ContextualAnalyzer,
        memory: MemoryRepository,
        *,
        llm_response_gen: Optional["LLMResponseGenerator"] = None,
        fallback_manager: Optional["FallbackManager"] = None,
        enable_llm: bool = False,
    ) -> None:
        self._rule_empathy = rule_empathy
        self._context_analyzer = context_analyzer
        self._memory = memory
        self._llm_response_gen = llm_response_gen
        self._fallback_manager = fallback_manager
        self._enable_llm = enable_llm

    def build(
        self,
        *,
        text: str,
        emotion: str,
        intensity: float,
        context: Optional[str],
        user_id: str,
        context_result: Dict,
    ) -> HumanResponse:
        """Generate empathy response and attach follow-up."""
        # Step 6: generate empathy response
        raw = self._generate_response(
            text=text,
            emotion=emotion,
            intensity=intensity,
            context=context,
            user_id=user_id,
        )

        # Step 7: attach follow-up suggestion from context analysis
        follow_up = self._context_analyzer.get_follow_up_suggestion(
            emotion, intensity, context_result,
        )
        if follow_up and not raw.follow_up:
            return HumanResponse(
                text=raw.text,
                empathy_type=raw.empathy_type,
                intensity_level=raw.intensity_level,
                follow_up=follow_up,
                empathy_depth=getattr(raw, "empathy_depth", "适度共情"),
                tone=getattr(raw, "tone", "温暖"),
            )

        return HumanResponse(
            text=raw.text,
            empathy_type=raw.empathy_type,
            intensity_level=raw.intensity_level,
            follow_up=raw.follow_up,
            empathy_depth=getattr(raw, "empathy_depth", "适度共情"),
            tone=getattr(raw, "tone", "温暖"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        text: str,
        emotion: str,
        intensity: float,
        context: Optional[str],
        user_id: str,
    ):
        """Generate empathy response, falling back from LLM to rules."""
        if not self._enable_llm or not self._llm_response_gen:
            return self._rule_empathy.generate(
                emotion=emotion,
                intensity=intensity,
                context=context,
            )
        try:
            profile = self._memory.get_user(user_id)
            history = (
                profile.emotional_history[-5:]
                if profile.emotional_history
                else []
            )
            return self._llm_response_gen.generate(
                text=text,
                emotion=emotion,
                intensity=intensity,
                context={"context": context} if context else None,
                user_profile={
                    "relationship_level": profile.relationship_level,
                    "total_interactions": profile.total_interactions,
                },
                conversation_history=history,
            )
        except Exception as e:
            logging.warning(
                "LLM response generation failed, falling back to rules: %s", e,
            )
            if self._fallback_manager:
                self._fallback_manager.record_failure(e)
            return self._rule_empathy.generate(
                emotion=emotion,
                intensity=intensity,
                context=context,
            )
