"""
Emotion analysis pipeline
================================
Steps 1-5 of the analysis flow:
  1. Emotion detection (LLM or rule engine)
  2. Primary emotion extraction
  3. VAD lookup
  4. Irony detection
  5. Context analysis + emotion classification
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from trueemotion.core.emotions.detector import HumanEmotionDetector
from trueemotion.core.emotions.plutchik24 import (
    EMOTION_VAD,
    get_intensity_label,
)
from trueemotion.core.emotions.irony import IronyDetector, IronyResult
from trueemotion.core.analysis.context import ContextualAnalyzer

# LLM types (optional)
try:
    from trueemotion.core.llm import LLMEmotionDetector, FallbackManager
    from trueemotion.core.llm.base import LLMError
except ImportError:
    LLMEmotionDetector = None
    FallbackManager = None
    LLMError = Exception


# Compound emotion names
_COMPOUND_EMOTIONS = frozenset({
    "bittersweet", "painful_joy", "happy_sadness",
    "gratitude_love", "hope_fear", "jealous_love",
    "frustration_hopelessness", "love_admiration",
    "anger_sadness", "shock_denial", "relief_sadness", "angry_fear",
})


@dataclass(frozen=True)
class PipelineResult:
    """Pipeline output for steps 1-5."""

    emotion_scores: Dict[str, float]
    primary_emotion: str
    primary_score: float
    vad: Tuple[float, float, float]
    confidence: float
    intensity_label: str
    explanation: Optional[Dict]
    irony_result: IronyResult
    context_result: Dict
    effective_emotion: str
    effective_intensity: float
    pure_emotions: Dict[str, float]
    compound_emotions: Dict[str, float]


class EmotionPipeline:
    """
    Runs steps 1-5 of the emotion analysis flow.

    Initialised once per EmotionAnalyzer lifetime,
    NOT per analyse call.
    """

    def __init__(
        self,
        rule_detector: HumanEmotionDetector,
        irony_detector: IronyDetector,
        context_analyzer: ContextualAnalyzer,
        *,
        llm_detector: Optional["LLMEmotionDetector"] = None,
        fallback_manager: Optional["FallbackManager"] = None,
        enable_llm: bool = False,
    ) -> None:
        self._rule_detector = rule_detector
        self._irony = irony_detector
        self._context_analyzer = context_analyzer
        self._llm_detector = llm_detector
        self._fallback_manager = fallback_manager
        self._enable_llm = enable_llm

    def run(self, text: str) -> PipelineResult:
        """Execute the pipeline and return a frozen result."""
        # 1. Emotion detection
        emotion_scores = self._detect_emotion(text)

        # 2. Primary emotion
        primary_emotion, primary_score = self._get_primary(emotion_scores)

        # 3. VAD + detailed LLM info
        vad, explanation, confidence = self._resolve_vad(
            text, primary_emotion, primary_score,
        )
        intensity_label = get_intensity_label(primary_score)

        # 4. Irony detection
        irony_result = self._irony.detect(text, primary_emotion, primary_score)
        effective_emotion = irony_result.true_emotion or primary_emotion
        effective_intensity = primary_score

        if irony_result.is_irony:
            effective_emotion = irony_result.true_emotion

        # 5. Context analysis + emotion classification
        context_result = self._context_analyzer.analyze_with_context(
            text, effective_emotion, effective_intensity,
        )
        pure_emotions = {
            k: v for k, v in emotion_scores.items()
            if k not in _COMPOUND_EMOTIONS
        }
        compound_emotions = {
            k: v for k, v in emotion_scores.items()
            if k in _COMPOUND_EMOTIONS
        }

        return PipelineResult(
            emotion_scores=emotion_scores,
            primary_emotion=primary_emotion,
            primary_score=primary_score,
            vad=vad,
            confidence=confidence,
            intensity_label=intensity_label,
            explanation=explanation,
            irony_result=irony_result,
            context_result=context_result,
            effective_emotion=effective_emotion,
            effective_intensity=effective_intensity,
            pure_emotions=pure_emotions,
            compound_emotions=compound_emotions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotion, falling back from LLM to rules."""
        if not self._enable_llm or not self._llm_detector:
            return self._rule_detector.detect(text)
        try:
            return self._llm_detector.detect(text)
        except Exception as e:
            logging.warning("LLM detection failed, falling back to rules: %s", e)
            if self._fallback_manager:
                self._fallback_manager.record_failure(e)
            return self._rule_detector.detect(text)

    @staticmethod
    def _get_primary(scores: Dict[str, float]) -> Tuple[str, float]:
        """Return (emotion, score) for the highest-scoring entry."""
        if not scores:
            return "neutral", 0.0
        primary = max(scores.items(), key=lambda x: x[1])
        return primary[0], primary[1]

    def _resolve_vad(
        self,
        text: str,
        primary_emotion: str,
        primary_score: float,
    ) -> Tuple[Tuple[float, float, float], Optional[Dict], float]:
        """Return (vad, explanation, confidence) from LLM or rule lookup."""
        if self._enable_llm and self._llm_detector:
            try:
                llm_result = self._llm_detector.get_detailed_result(text)
                vad = (
                    llm_result.get("vad", {}).get("valence", 0.0),
                    llm_result.get("vad", {}).get("arousal", 0.0),
                    llm_result.get("vad", {}).get("dominance", 0.0),
                )
                explanation = llm_result.get("explanation")
                confidence = llm_result.get("confidence", primary_score)
                return vad, explanation, confidence
            except (LLMError, RuntimeError, KeyError, TypeError) as e:
                logging.warning(
                    "LLM detailed result failed, using rule fallback: %s", e,
                )

        vad = EMOTION_VAD.get(primary_emotion, (0.0, 0.0, 0.0))
        return vad, None, primary_score
