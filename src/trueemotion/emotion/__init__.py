# -*- coding: utf-8 -*-
"""
TrueEmotion Emotion Package
"""

from trueemotion.emotion.plutchik24 import (
    EMOTION_DEFINITIONS,
    PrimaryEmotion,
    ComplexEmotion,
    get_primary_emotions,
    get_complex_emotions,
    get_all_emotions,
    get_emotion_by_name,
    is_primary_emotion,
    is_complex_emotion,
    VAD_LEXICON
)

from trueemotion.emotion.emotion_output import (
    EmotionOutput,
    EmotionContext,
    EmotionSample
)

__all__ = [
    "EMOTION_DEFINITIONS",
    "PrimaryEmotion",
    "ComplexEmotion",
    "get_primary_emotions",
    "get_complex_emotions",
    "get_all_emotions",
    "get_emotion_by_name",
    "is_primary_emotion",
    "is_complex_emotion",
    "VAD_LEXICON",
    "EmotionOutput",
    "EmotionContext",
    "EmotionSample",
]
