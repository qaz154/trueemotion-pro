"""
情感名称国际化（中文映射）
===========================
统一的 emotion -> 中文名 映射，合并自 analyzer 和 engine 的重复定义
"""

from typing import Dict

__all__ = ["EMOTION_CN"]

# 统一情感中文名映射
# 合并自 analyzer._get_emotion_cn 与 engine._substitute_variables
EMOTION_CN: Dict[str, str] = {
    # 基础情感
    "joy": "喜悦",
    "sadness": "悲伤",
    "anger": "愤怒",
    "fear": "恐惧",
    "disgust": "厌恶",
    "surprise": "惊讶",
    "trust": "信任",
    "anticipation": "期待",
    # 强度变体
    "ecstasy": "狂喜",
    "grief": "悲痛",
    "rage": "暴怒",
    "terror": "恐惧",
    # 复杂情感
    "anxiety": "焦虑",
    "love": "爱",
    "hope": "希望",
    "despair": "绝望",
    "guilt": "内疚",
    "pride": "自豪",
    "envy": "嫉妒",
    "contempt": "鄙视",
    "boredom": "无聊",
    "loneliness": "孤独",
    "compassion": "同情",
    "gratitude": "感激",
    "regret": "遗憾",
    "confusion": "困惑",
    "nostalgia": "怀旧",
    "contentment": "满足",
    "melancholy": "忧郁",
    # 复合情感
    "bittersweet": "悲喜交加",
    # 特殊
    "neutral": "中性",
}
