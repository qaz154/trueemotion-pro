"""
TrueEmotion Pro - 新一代中文情感AI系统 v4.0
"""

__version__ = "4.0.0"
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
from trueemotion.core.emotions.detector import RuleBasedEmotionDetector

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
    "RuleBasedEmotionDetector",
]
