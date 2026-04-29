"""
情感分析输出数据结构
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EmotionOutput:
    """单条情感输出"""
    primary: str
    intensity: float
    vad: tuple[float, float, float]
    confidence: float
    secondary: Optional[str] = None
    all_emotions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanResponse:
    """人性化回复"""
    text: str
    empathy_type: str  # support, comfort, excitement, etc.
    intensity_level: str  # extreme, high, moderate, low, minimal
    follow_up: Optional[str] = None


@dataclass(frozen=True)
class UserProfile:
    """用户画像"""
    user_id: str
    total_interactions: int = 0
    dominant_emotion: Optional[str] = None
    relationship_level: float = 0.0
    learned_patterns: int = 0
    last_emotion: Optional[str] = None
    emotional_history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisResult:
    """完整分析结果"""
    version: str
    engine: str
    emotion: EmotionOutput
    human_response: HumanResponse
    user_profile: UserProfile
    context_used: bool = False
