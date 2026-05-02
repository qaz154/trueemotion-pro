"""
分析结果数据结构 v1.15
========================
支持复合情感、连续强度等新特性
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass(frozen=True)
class EmotionOutput:
    """
    情感输出

    核心变化:
    - primary: 主要情感（原来只有一个，现在可能是复合情感）
    - all_emotions: 所有检测到的情感及其连续强度
    - compound_emotions: 复合情感（如悲喜交加）
    - intensity: 连续强度分数 0.0-1.0
    - vad: Valence-Arousal-Dominance 坐标
    """
    primary: str
    intensity: float
    vad: Tuple[float, float, float]
    confidence: float
    intensity_label: str = "中等"  # 新增：人类可读强度
    all_emotions: Dict[str, float] = field(default_factory=dict)  # 新增：所有情感
    compound_emotions: Dict[str, float] = field(default_factory=dict)  # 新增：复合情感
    emotion_mix: List[Tuple[str, float]] = field(default_factory=list)  # 新增：情感混合描述


@dataclass(frozen=True)
class HumanResponse:
    """
    人性化回复

    新增:
    - empathy_depth: 共情深度描述
    - tone: 语气描述
    - adaptation_notes: 调整说明
    """
    text: str
    empathy_type: str
    intensity_level: str
    follow_up: Optional[str] = None
    empathy_depth: str = "适度共情"  # 新增
    tone: str = "温暖"  # 新增


@dataclass(frozen=True)
class UserProfile:
    """用户画像"""
    user_id: str
    total_interactions: int = 0
    dominant_emotion: Optional[str] = None
    relationship_level: float = 0.0
    learned_patterns: int = 0
    last_emotion: Optional[str] = None
    emotional_history: List[str] = field(default_factory=list)
    emotional_state: str = "平稳"  # 新增：当前情绪状态
    interaction_style: str = "正常"  # 新增：互动风格


@dataclass(frozen=True)
class AnalysisResult:
    """
    完整分析结果 v1.15

    新增字段:
    - emotion_mix: 情感混合
    - personality: 个性化信息
    - context_analysis: 上下文分析
    """
    version: str
    engine: str
    emotion: EmotionOutput
    human_response: HumanResponse
    user_profile: UserProfile
    context_used: bool = False
    emotion_mix: str = ""  # 新增：如"以悲伤为主，伴有轻微愤怒"
    explanation: Optional[Dict] = None  # 新增：检测解释
