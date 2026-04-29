# -*- coding: utf-8 -*-
"""
Context Encoder Module
====================

上下文感知编码器：追踪对话历史，理解情感演变

核心设计：
1. 滑动窗口维护对话历史
2. 情感状态追踪
3. 上下文相关情感理解
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Utterance:
    """对话片段"""
    speaker: str  # "user", "agent", "system"
    text: str
    emotion: Optional[str] = None
    intensity: float = 0.0
    timestamp: Optional[str] = None

    def __str__(self) -> str:
        emotion_str = f"[{self.emotion}]" if self.emotion else ""
        return f"{self.speaker}: {self.text}{emotion_str}"


@dataclass
class ContextWindow:
    """
    对话上下文窗口

    维护最近N轮对话，用于情感理解
    """
    window_size: int = 5
    utterances: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, utterance: Utterance) -> None:
        """添加对话片段"""
        self.utterances.append(utterance)

    def get_recent(self, n: int = 3) -> List[Utterance]:
        """获取最近N轮对话"""
        return list(self.utterances)[-n:]

    def get_history_text(self) -> str:
        """获取历史对话文本"""
        return "\n".join(str(u) for u in self.utterances)

    def get_user_history(self) -> List[str]:
        """获取用户历史发言"""
        return [u.text for u in self.utterances if u.speaker == "user"]

    def get_agent_history(self) -> List[str]:
        """获取AI历史发言"""
        return [u.text for u in self.utterances if u.speaker == "agent"]

    def get_emotion_history(self) -> List[Tuple[str, float]]:
        """获取历史情感"""
        return [(u.emotion, u.intensity) for u in self.utterances if u.emotion]

    def is_emotion_escalating(self) -> bool:
        """检测情感是否在升级"""
        emotions = self.get_emotion_history()
        if len(emotions) < 2:
            return False

        # 检查最近2-3轮是否持续升级
        recent = emotions[-3:]
        intensities = [e[1] for e in recent]
        return all(intensities[i] <= intensities[i+1] for i in range(len(intensities)-1))

    def is_emotion_repeating(self) -> bool:
        """检测情感是否重复（持续同一情感）"""
        emotions = self.get_emotion_history()
        if len(emotions) < 3:
            return False

        recent = emotions[-3:]
        emotion_names = [e[0] for e in recent]
        return len(set(emotion_names)) == 1


class ContextEncoder:
    """
    上下文感知编码器

    核心功能：
    1. 编码对话历史为上下文向量
    2. 识别情感演变模式
    3. 提供上下文特征供情感分析使用
    """

    def __init__(self, window_size: int = 5):
        self.window = ContextWindow(window_size=window_size)

        # 情感对比模式
        self.EMOTION_CONTRASTS = {
            "joy": ["sadness", "anger"],
            "sadness": ["joy", "anger"],
            "anger": ["joy", "sadness"],
            "fear": ["joy", "anticipation"],
            "surprise": ["sadness", "joy"],
        }

        # 情感升级标记词
        self.ESCALATION_MARKERS = [
            "更", "越来越", "越来越", "真是", "简直",
            "居然", "竟然", "怎么", "为什么"
        ]

        # 情感对比标记词
        self.CONTRAST_MARKERS = [
            "但是", "可是", "然而", "不过", "却",
            "反而", "只是", "虽然", "尽管"
        ]

    def add_utterance(self, text: str, speaker: str = "user",
                     emotion: Optional[str] = None, intensity: float = 0.0) -> None:
        """添加对话片段"""
        utterance = Utterance(
            speaker=speaker,
            text=text,
            emotion=emotion,
            intensity=intensity
        )
        self.window.add(utterance)

    def encode(self, current_text: str) -> Dict[str, Any]:
        """
        编码当前文本的上下文特征

        Returns:
            Dict包含:
            - context_features: 上下文特征字典
            - context_text: 上下文文本
            - emotion_patterns: 检测到的情感模式
        """
        features = {}

        # 1. 历史统计特征
        features["history_length"] = len(self.window.utterances)
        features["user_turn_count"] = len(self.window.get_user_history())
        features["agent_turn_count"] = len(self.window.get_agent_history())

        # 2. 情感演变特征
        emotion_history = self.window.get_emotion_history()
        if emotion_history:
            recent_emotions = [e[0] for e in emotion_history[-3:]]
            features["recent_emotions"] = recent_emotions
            features["is_escalating"] = self.window.is_emotion_escalating()
            features["is_repeating"] = self.window.is_emotion_repeating()
            features["emotion_diversity"] = len(set(recent_emotions))
        else:
            features["recent_emotions"] = []
            features["is_escalating"] = False
            features["is_repeating"] = False
            features["emotion_diversity"] = 0

        # 3. 情感对比检测
        features["has_contrast"] = self._has_contrast_marker(current_text)
        features["contrast_type"] = self._detect_contrast_type(current_text)

        # 4. 情感升级检测
        features["has_escalation"] = self._has_escalation_marker(current_text)
        features["escalation_type"] = self._detect_escalation_type(current_text)

        # 5. 上下文情感权重
        features["context_emotion_weight"] = self._calculate_context_weight()

        # 6. 用户历史情感摘要
        features["user_emotion_summary"] = self._summarize_user_emotions()

        # 7. 最近一轮的情感
        recent = self.window.get_recent(1)
        if recent:
            features["last_emotion"] = recent[-1].emotion
            features["last_intensity"] = recent[-1].intensity
        else:
            features["last_emotion"] = None
            features["last_intensity"] = 0.0

        # 8. 生成上下文文本（供模型使用）
        context_text = self._build_context_text(current_text)

        return {
            "features": features,
            "context_text": context_text,
            "emotion_patterns": self._detect_emotion_patterns(current_text)
        }

    def _has_contrast_marker(self, text: str) -> bool:
        """检测是否有对比标记"""
        return any(marker in text for marker in self.CONTRAST_MARKERS)

    def _detect_contrast_type(self, text: str) -> Optional[str]:
        """检测对比类型"""
        if "但是" in text or "可是" in text:
            return "but"
        elif "然而" in text:
            return "however"
        elif "却" in text:
            return "contradiction"
        elif "反而" in text:
            return "reverse"
        elif "虽然" in text or "尽管" in text:
            return "concession"
        return None

    def _has_escalation_marker(self, text: str) -> bool:
        """检测是否有升级标记"""
        return any(marker in text for marker in self.ESCALATION_MARKERS)

    def _detect_escalation_type(self, text: str) -> Optional[str]:
        """检测升级类型"""
        if "更" in text:
            return "increasing"
        elif "越来越" in text:
            return "progressive"
        elif "真是" in text or "简直" in text:
            return "exaggeration"
        elif "居然" in text or "竟然" in text:
            return "unexpected"
        elif "怎么" in text or "为什么" in text:
            return "questioning"
        return None

    def _calculate_context_weight(self) -> float:
        """计算上下文对当前情感判断的影响权重"""
        weight = 0.3  # 基础权重

        # 如果情感正在升级，提高权重
        if self.window.is_emotion_escalating():
            weight += 0.2

        # 如果情感在重复，提高权重（持续的情感更重要）
        if self.window.is_emotion_repeating():
            weight += 0.15

        # 如果历史较长，降低权重（太远的上下文不太相关）
        history_len = len(self.window.utterances)
        if history_len > 3:
            weight -= (history_len - 3) * 0.05

        return max(0.1, min(0.5, weight))

    def _summarize_user_emotions(self) -> Dict[str, float]:
        """总结用户历史情感"""
        emotions = [e[0] for e in self.window.get_emotion_history() if e[0]]
        if not emotions:
            return {}

        from collections import Counter
        counts = Counter(emotions)
        total = len(emotions)

        return {
            emotion: count / total
            for emotion, count in counts.most_common(3)
        }

    def _build_context_text(self, current_text: str) -> str:
        """构建上下文字符串"""
        parts = []

        # 添加历史对话
        history = self.window.get_recent(3)
        if history:
            parts.append("对话历史:")
            for u in history:
                parts.append(f"  {str(u)}")

        # 添加当前文本
        parts.append(f"当前: {current_text}")

        return "\n".join(parts)

    def _detect_emotion_patterns(self, current_text: str) -> List[str]:
        """检测情感模式"""
        patterns = []

        # 1. 对比模式
        if self._has_contrast_marker(current_text):
            patterns.append("contrast")

            # 检查是否从正面转向负面
            recent_emotions = self.window.get_emotion_history()
            if recent_emotions:
                last_emotion = recent_emotions[-1][0] if recent_emotions else None
                if last_emotion in ["joy", "anticipation"]:
                    patterns.append("positive_to_negative")

        # 2. 升级模式
        if self._has_escalation_marker(current_text):
            patterns.append("escalation")

        # 3. 重复模式
        if self.window.is_emotion_repeating():
            patterns.append("emotion_persistence")

        # 4. 累积模式（连续多个短句）
        user_history = self.window.get_user_history()
        if len(user_history) >= 2 and all(len(t) < 15 for t in user_history[-2:]):
            patterns.append("rapid_fire")

        # 5. 问答模式（用户提问）
        if "?" in current_text or "？" in current_text:
            patterns.append("question")

        # 6. 感叹模式
        if "!" in current_text or "！" in current_text:
            patterns.append("exclamation")

        return patterns

    def get_context_for_prompt(self, current_text: str) -> str:
        """
        获取用于情感分析的上下文提示

        这是一个简化版本，供基于LLM的情感分析使用
        """
        history = self.window.get_recent(3)
        if not history:
            return current_text

        parts = ["[对话上下文]\n"]
        for u in history:
            parts.append(f"-{str(u)}")
        parts.append(f"\n[当前发言]\n-{current_text}")

        return "\n".join(parts)

    def reset(self) -> None:
        """重置上下文"""
        self.window = ContextWindow(window_size=self.window.window_size)


class EmotionPatternMatcher:
    """
    情感模式匹配器

    基于规则匹配情感相关模式
    """

    # 强度修饰词
    INTENSITY_MODIFIERS = {
        "极其": 1.3, "非常": 1.2, "特别": 1.2, "十分": 1.2,
        "很": 1.1, "比较": 1.0, "有点": 0.8, "稍微": 0.7,
        "略微": 0.6, "一点点": 0.5
    }

    # 否定词
    NEGATIONS = ["不", "没", "无", "非", "别", "休", "未"]

    # 情感词后缀
    EMOTION_SUFFIXES = ["的", "地", "得", "了", "啊", "呀", "吧", "哦", "呢", "哈"]

    @classmethod
    def extract_intensity_modifier(cls, text: str) -> Tuple[float, str]:
        """提取强度修饰词"""
        for word, multiplier in cls.INTENSITY_MODIFIERS.items():
            if word in text:
                return multiplier, word
        return 1.0, ""

    @classmethod
    def has_negation(cls, text: str) -> bool:
        """检查是否有否定词"""
        # 简单检查，实际需要更复杂的否定范围检测
        for neg in cls.NEGATIONS:
            if neg in text:
                return True
        return False

    @classmethod
    def extract_emotion_keywords(cls, text: str, emotion_keywords: Dict[str, List[str]]) -> Dict[str, int]:
        """提取情感关键词及其计数"""
        result = {}
        text_lower = text.lower()

        for emotion, keywords in emotion_keywords.items():
            count = 0
            for kw in keywords:
                if kw in text_lower:
                    count += text_lower.count(kw)
            if count > 0:
                result[emotion] = count

        return result

    @classmethod
    def detect_mixed_emotion(cls, text: str, emotion_keywords: Dict[str, List[str]]) -> List[str]:
        """检测混合情感"""
        found = cls.extract_emotion_keywords(text, emotion_keywords)
        if len(found) >= 2:
            return sorted(found.keys(), key=lambda x: found[x], reverse=True)[:3]
        return []


if __name__ == "__main__":
    # 测试上下文编码器
    encoder = ContextEncoder(window_size=5)

    # 模拟对话
    dialogues = [
        ("我今天加班到很晚，好累啊。", "user", "sadness", 0.6),
        ("辛苦了，工作不要太拼。", "agent", "trust", 0.4),
        ("可是项目还是没完成...", "user", "anxiety", 0.7),
        ("别着急，慢慢来。", "agent", "trust", 0.3),
        ("唉，明天还要汇报...", "user", "fear", 0.65),
    ]

    for text, speaker, emotion, intensity in dialogues:
        encoder.add_utterance(text, speaker, emotion, intensity)

    # 测试编码
    test_text = "怎么办啊，真的很焦虑！"
    result = encoder.encode(test_text)

    print("=" * 60)
    print("上下文编码测试")
    print("=" * 60)
    print(f"\n当前文本: {test_text}")
    print(f"\n上下文特征:")
    for key, value in result["features"].items():
        print(f"  {key}: {value}")

    print(f"\n检测到的情感模式: {result['emotion_patterns']}")

    print(f"\n上下文字符串:")
    print(result["context_text"])
