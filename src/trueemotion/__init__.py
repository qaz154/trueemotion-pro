"""
TrueEmotion Pro - 人性化情感AI系统
让AI拥有像人类一样丰富、复杂、真实的情感
"""

from trueemotion._version import __version__

__author__ = "TrueEmotion Team"

from trueemotion.api.facade import TrueEmotionPro, create_analyzer
from trueemotion.core.emotions.detector import HumanEmotionDetector

__all__ = [
    "__version__",
    "TrueEmotionPro",
    "create_analyzer",
    "HumanEmotionDetector",
]
