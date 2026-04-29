# -*- coding: utf-8 -*-
"""
Emotion Output Data Structure
=============================

TrueEmotion的统一输出格式：支持多标签情感、强度、VAD、反讽检测
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


@dataclass
class EmotionOutput:
    """
    情感分析统一输出

    支持：
    - 多标签情感分类（原始+复合）
    - 情感强度（0-1连续值）
    - VAD维度连续值
    - 反讽/隐喻检测
    - 置信度评估
    """

    # 原型情感强度 {emotion_name: intensity} - 8种原始情感
    primary: Dict[str, float] = field(default_factory=dict)

    # 复合情感标签 {emotion_name: bool} - 16种复合情感
    complex: Dict[str, bool] = field(default_factory=dict)

    # VAD维度 (Valence, Arousal, Dominance) - 范围[-1, 1]
    vad: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 情感强度 [0, 1] - 0=无感, 1=极度
    intensity: float = 0.0

    # 反讽检测
    is_irony: bool = False
    irony_confidence: float = 0.0  # 反讽置信度
    surface_emotion: Optional[str] = None  # 表面情感
    true_emotion: Optional[str] = None  # 真实情感

    # 隐喻检测
    is_metaphor: bool = False
    metaphor_confidence: float = 0.0

    # 模型置信度 [0, 1]
    confidence: float = 0.0

    # 情感状态
    state: str = "BASELINE"  # BASELINE, ACTIVE, DECAYING, SUPPRESSED

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def get_primary_emotion(self) -> Optional[str]:
        """获取最强烈的原型情感"""
        if not self.primary:
            return None
        return max(self.primary.items(), key=lambda x: x[1])[0]

    def get_primary_intensity(self) -> float:
        """获取最强烈原型情感的强度"""
        primary_emotion = self.get_primary_emotion()
        if primary_emotion:
            return self.primary.get(primary_emotion, 0.0)
        return 0.0

    def get_complex_emotions(self) -> List[str]:
        """获取所有被激活的复合情感"""
        return [k for k, v in self.complex.items() if v]

    def has_emotion(self, emotion: str) -> bool:
        """检查是否有指定情感"""
        emotion = emotion.lower()
        if emotion in self.primary:
            return self.primary[emotion] > 0.3
        if emotion in self.complex:
            return self.complex[emotion]
        return False

    def is_positive(self) -> bool:
        """判断整体情感是否为正面"""
        return self.vad[0] > 0.2

    def is_negative(self) -> bool:
        """判断整体情感是否为负面"""
        return self.vad[0] < -0.2

    def is_high_arousal(self) -> bool:
        """判断是否为高唤醒状态"""
        return self.vad[1] > 0.3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "primary": self.primary,
            "complex": self.complex,
            "vad": {"valence": self.vad[0], "arousal": self.vad[1], "dominance": self.vad[2]},
            "intensity": self.intensity,
            "is_irony": self.is_irony,
            "irony_confidence": self.irony_confidence,
            "surface_emotion": self.surface_emotion,
            "true_emotion": self.true_emotion,
            "is_metaphor": self.is_metaphor,
            "metaphor_confidence": self.metaphor_confidence,
            "confidence": self.confidence,
            "state": self.state,
            "primary_emotion": self.get_primary_emotion(),
            "complex_emotions": self.get_complex_emotions(),
            "is_positive": self.is_positive(),
            "is_negative": self.is_negative(),
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """友好字符串表示"""
        primary = self.get_primary_emotion() or "none"
        intensity_pct = int(self.intensity * 100)
        vad_str = f"V={self.vad[0]:.2f},A={self.vad[1]:.2f},D={self.vad[2]:.2f}"

        parts = [f"{primary} ({intensity_pct}%)"]

        if self.complex:
            complex_list = self.get_complex_emotions()
            if complex_list:
                parts.append(f"complex=[{','.join(complex_list[:3])}]")

        parts.append(vad_str)

        if self.is_irony:
            parts.append(f"IRONY(conf={self.irony_confidence:.2f})")

        if self.confidence < 0.5:
            parts.append(f"LOW_CONF({self.confidence:.2f})")

        return " | ".join(parts)


@dataclass
class EmotionContext:
    """
    情感上下文 - 追踪对话历史中的情感变化
    """

    # 对话历史窗口
    window_size: int = 5

    # 历史情感记录
    history: List[EmotionOutput] = field(default_factory=list)

    # 当前情感状态
    current_state: str = "BASELINE"

    # 活跃情感栈
    active_emotions: Dict[str, float] = field(default_factory=dict)

    def add(self, emotion: EmotionOutput) -> None:
        """添加新的情感记录"""
        self.history.append(emotion)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        self._update_active_emotions(emotion)

    def _update_active_emotions(self, emotion: EmotionOutput) -> None:
        """更新活跃情感"""
        for name, intensity in emotion.primary.items():
            if intensity > 0.5:
                self.active_emotions[name] = max(
                    self.active_emotions.get(name, 0.0),
                    intensity
                )

        # 衰减旧情感
        decay_rate = 0.2
        for name in list(self.active_emotions.keys()):
            self.active_emotions[name] *= (1 - decay_rate)
            if self.active_emotions[name] < 0.1:
                del self.active_emotions[name]

    def get_recent_emotions(self) -> List[str]:
        """获取最近的情感列表"""
        return [e.get_primary_emotion() for e in self.history[-3:] if e.get_primary_emotion()]

    def has_emotion_trend(self, emotion: str) -> bool:
        """检查情感趋势（连续出现）"""
        recent = self.get_recent_emotions()
        if len(recent) < 2:
            return False
        return all(e == emotion for e in recent[-2:])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "window_size": self.window_size,
            "history_count": len(self.history),
            "current_state": self.current_state,
            "active_emotions": self.active_emotions,
            "recent_emotions": self.get_recent_emotions(),
            "history": [h.to_dict() for h in self.history[-3:]]
        }


@dataclass
class EmotionSample:
    """
    训练样本格式
    """

    # 输入文本
    text: str

    # 场景ID
    scenario_id: str

    # 原型情感标签 {emotion: intensity}
    primary_labels: Dict[str, float] = field(default_factory=dict)

    # 复合情感标签 {emotion: bool}
    complex_labels: Dict[str, bool] = field(default_factory=dict)

    # VAD标签
    vad_labels: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 强度标签
    intensity_label: float = 0.0

    # 反讽标签
    is_irony: bool = False
    surface_emotion: Optional[str] = None
    true_emotion: Optional[str] = None

    # 上下文（可选）
    context: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "scenario_id": self.scenario_id,
            "primary_labels": self.primary_labels,
            "complex_labels": self.complex_labels,
            "vad_labels": {
                "valence": self.vad_labels[0],
                "arousal": self.vad_labels[1],
                "dominance": self.vad_labels[2]
            },
            "intensity_label": self.intensity_label,
            "is_irony": self.is_irony,
            "surface_emotion": self.surface_emotion,
            "true_emotion": self.true_emotion,
            "context": self.context
        }
