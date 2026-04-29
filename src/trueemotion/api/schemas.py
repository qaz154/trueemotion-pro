"""
API数据模型
使用Pydantic风格的数据类定义
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalyzeRequest:
    """分析请求"""
    text: str
    user_id: str = "default"
    learn: bool = False
    response: Optional[str] = None
    feedback: float = 0.5
    context: Optional[str] = None


@dataclass
class EmotionResult:
    """情感分析结果"""
    version: str
    engine: str
    emotion: "EmotionData"
    human_response: "ResponseData"
    user_profile: "ProfileData"
    context_used: bool = False


@dataclass
class EmotionData:
    """情感数据"""
    primary: str
    intensity: float
    vad: tuple[float, float, float]
    confidence: float
    all_emotions: dict[str, float] = field(default_factory=dict)


@dataclass
class ResponseData:
    """回复数据"""
    text: str
    empathy_type: str
    intensity_level: str
    follow_up: Optional[str] = None


@dataclass
class ProfileData:
    """用户画像数据"""
    user_id: str
    total_interactions: int = 0
    dominant_emotion: Optional[str] = None
    relationship_level: float = 0.0
    learned_patterns: int = 0
    last_emotion: Optional[str] = None


@dataclass
class EvolutionResult:
    """进化结果"""
    total_patterns_analyzed: int
    emotions_with_patterns: int
    evolved_rules: list[dict]
    evolution_version: str


@dataclass
class SystemStats:
    """系统统计"""
    total_users: int
    total_patterns: int
    memory_path: str
    version: str
