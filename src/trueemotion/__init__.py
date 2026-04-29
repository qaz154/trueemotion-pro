# TrueEmotion - 新一代真实情感AI系统

from trueemotion.models.emotion_model import EmotionAnalyzer, TrueEmotionModel
from trueemotion.emotion.plutchik24 import (
    EMOTION_DEFINITIONS,
    get_primary_emotions,
    get_complex_emotions,
    get_all_emotions
)
from trueemotion.emotion.emotion_output import EmotionOutput, EmotionContext

__version__ = "1.0.0"
__all__ = [
    "EmotionAnalyzer",
    "TrueEmotionModel",
    "EmotionOutput",
    "EmotionContext",
    "EMOTION_DEFINITIONS",
    "get_primary_emotions",
    "get_complex_emotions",
    "get_all_emotions",
]
