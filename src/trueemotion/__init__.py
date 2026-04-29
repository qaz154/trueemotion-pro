"""
TrueEmotion Pro v1.12 - 人性化情感AI系统
让AI拥有像人类一样丰富、复杂、真实的情感
"""

__version__ = "1.12"
__author__ = "TrueEmotion Team"

from trueemotion.api.routes import TrueEmotionPro, create_analyzer
from trueemotion.api.schemas import (
    AnalyzeRequest,
    EmotionResult,
    EmotionData,
    ResponseData,
    ProfileData,
    EvolutionResult,
    SystemStats,
)
from trueemotion.core.emotions.detector import HumanEmotionDetector

__all__ = [
    # Main API
    "TrueEmotionPro",
    "create_analyzer",
    # Schemas
    "AnalyzeRequest",
    "EmotionResult",
    "EmotionData",
    "ResponseData",
    "ProfileData",
    "EvolutionResult",
    "SystemStats",
    # Core
    "HumanEmotionDetector",
]
