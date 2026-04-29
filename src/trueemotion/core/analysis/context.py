"""
上下文理解系统 v1.13
====================
让AI能感知对话历史，实现情感连贯

核心理念:
1. 对话流畅 - 不孤立看单句，理解情感变化
2. 话题追踪 - 识别当前讨论的话题
3. 情感趋势 - 感知用户情绪是好转还是恶化
4. 关系发展 - 了解对话双方的亲密度变化
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from collections import deque


@dataclass
class Utterance:
    """对话片段"""
    text: str
    emotion: str
    intensity: float
    timestamp: datetime = field(default_factory=datetime.now)
    is_user: bool = True  # True=用户, False=AI


@dataclass
class EmotionTrend:
    """情感趋势"""
    direction: str  # "rising", "falling", "stable"
    delta: float  # 变化量
    dominant_change: str  # 主要变化


class ConversationContext:
    """
    对话上下文

    维护最近N轮对话的历史，理解情感变化趋势
    """

    MAX_HISTORY = 10  # 保留最近10轮

    def __init__(self):
        self.history: deque[Utterance] = deque(maxlen=self.MAX_HISTORY)
        self.current_topic: Optional[str] = None
        self.topic_history: List[str] = []
        self.session_start: datetime = datetime.now()

    def add(
        self,
        text: str,
        emotion: str,
        intensity: float,
        is_user: bool = True,
    ) -> None:
        """添加一轮对话"""
        utterance = Utterance(
            text=text,
            emotion=emotion,
            intensity=intensity,
            is_user=is_user,
        )
        self.history.append(utterance)

    def get_recent_emotions(self, n: int = 3) -> List[str]:
        """获取最近N轮的用户情感"""
        user_utterances = [u for u in self.history if u.is_user]
        return [u.emotion for u in user_utterances[-n:]]

    def get_emotion_trend(self) -> EmotionTrend:
        """分析情感趋势"""
        user_utterances = [u for u in self.history if u.is_user]

        if len(user_utterances) < 2:
            return EmotionTrend("stable", 0.0, "insufficient_data")

        intensities = [u.intensity for u in user_utterances[-5:]]
        emotions = [u.emotion for u in user_utterances[-5:]]

        # 计算趋势
        if len(intensities) >= 2:
            delta = intensities[-1] - intensities[0]

            if delta > 0.1:
                direction = "rising"
            elif delta < -0.1:
                direction = "falling"
            else:
                direction = "stable"

            # 检测主要情感变化
            if len(emotions) >= 2 and emotions[-1] != emotions[0]:
                dominant_change = f"{emotions[0]}→{emotions[-1]}"
            else:
                dominant_change = emotions[-1] if emotions else "neutral"

            return EmotionTrend(direction, delta, dominant_change)

        return EmotionTrend("stable", 0.0, "neutral")

    def get_context_summary(self) -> Dict:
        """获取上下文摘要"""
        trend = self.get_emotion_trend()
        recent = self.get_recent_emotions(3)

        return {
            "turns": len([u for u in self.history if u.is_user]),
            "recent_emotions": recent,
            "trend": trend.direction,
            "trend_delta": round(trend.delta, 3),
            "main_change": trend.dominant_change,
            "session_duration_seconds": (datetime.now() - self.session_start).seconds,
        }

    def was_emotion_mentioned_recently(self, emotion: str, within: int = 3) -> bool:
        """检查某种情感是否在最近N轮中出现过"""
        recent_emotions = self.get_recent_emotions(within)
        return emotion in recent_emotions

    def get_last_user_emotion(self) -> Optional[Tuple[str, float]]:
        """获取上一轮用户情感"""
        for u in reversed(self.history):
            if u.is_user:
                return (u.emotion, u.intensity)
        return None

    def clear(self) -> None:
        """清空上下文"""
        self.history.clear()
        self.current_topic = None


class ContextualAnalyzer:
    """
    上下文感知分析器

    在普通分析基础上，结合对话历史给出更准确的判断
    """

    # 情感变化模式
    EMOTION_TRANSITIONS = {
        # 负面加强
        ("sadness", "sadness"): "reinforced",
        ("fear", "fear"): "reinforced",
        ("anger", "anger"): "reinforced",
        # 正面减弱
        ("joy", "sadness"): "turning_worse",
        ("hope", "fear"): "worsening",
        # 负面转正面
        ("sadness", "joy"): "improving",
        ("fear", "relief"): "recovering",
        ("anger", "calm"): "cooling_down",
        # 持续正面
        ("joy", "joy"): "sustained_positive",
        ("love", "joy"): "deepening_positive",
    }

    def __init__(self):
        self._context = ConversationContext()

    def analyze_with_context(
        self,
        text: str,
        base_emotion: str,
        base_intensity: float,
    ) -> Dict:
        """
        结合上下文的分析

        Args:
            text: 当前文本
            base_emotion: 基础情感分析结果
            base_intensity: 基础强度

        Returns:
            Dict: 包含上下文调整后的分析
        """
        last = self._context.get_last_user_emotion()

        # 检测情感变化
        context_adjustment = "new"
        if last:
            last_emotion, last_intensity = last
            transition_key = (last_emotion, base_emotion)
            context_adjustment = self.EMOTION_TRANSITIONS.get(
                transition_key,
                "continuing" if last_emotion == base_emotion else "shifting"
            )

        # 情感趋势
        trend = self._context.get_emotion_trend()

        # 更新上下文
        self._context.add(text, base_emotion, base_intensity)

        return {
            "base_emotion": base_emotion,
            "base_intensity": base_intensity,
            "context_adjustment": context_adjustment,
            "trend": trend.direction,
            "is_repeated_emotion": self._context.was_emotion_mentioned_recently(
                base_emotion, within=2
            ),
            "needs_acknowledgment": context_adjustment == "reinforced",
        }

    def get_follow_up_suggestion(
        self,
        base_emotion: str,
        base_intensity: float,
        context_analysis: Dict,
    ) -> Optional[str]:
        """基于上下文生成追问建议"""
        adjustment = context_analysis.get("context_adjustment", "new")
        trend = context_analysis.get("trend", "stable")
        is_repeated = context_analysis.get("is_repeated_emotion", False)

        # 如果情绪被强化，需要确认/深入
        if adjustment == "reinforced" and base_intensity > 0.3:
            return self._get_deepening_question(base_emotion)

        # 如果情绪在恶化，需要关心
        if trend == "falling" and base_intensity > 0.4:
            return self._get_caring_question(base_emotion)

        # 如果是持续相同情绪
        if is_repeated and base_intensity > 0.5:
            return self._get_acknowledgment(base_emotion)

        return None

    def _get_deepening_question(self, emotion: str) -> str:
        """生成深入性问题"""
        deepeners = {
            "sadness": ["还在想那件事吗？", "说出来会好受点"],
            "fear": ["还在担心吗？", "能说说具体在怕什么吗？"],
            "anger": ["还生气吗？", "能说说是什么让你这么生气吗？"],
            "joy": ["还在回味那件事吗？", "太棒了，详细讲讲！"],
            "anxiety": ["还在担心吗？", "能说说具体是什么情况吗？"],
        }
        import random
        questions = deepeners.get(emotion, ["后来呢？", "然后呢？"])
        return random.choice(questions)

    def _get_caring_question(self, emotion: str) -> str:
        """生成关心性问题"""
        caring = {
            "sadness": "我能帮你做点什么吗？",
            "fear": "需要我陪你说说话吗？",
            "anxiety": "深呼吸，我们一起看看怎么解决",
            "despair": "先休息一下，我在这里",
        }
        return caring.get(emotion, "怎么了？还好吗？")

    def _get_acknowledgment(self, emotion: str) -> str:
        """生成确认性回复"""
        acknowledgments = {
            "sadness": "还在难过啊，我陪着你",
            "fear": "还在担心是吧，我听着",
            "anger": "还是气不过吗？",
            "joy": "还是这么开心！发生什么好事了？",
        }
        return acknowledgments.get(emotion, "嗯，我听着")

    def get_context(self) -> ConversationContext:
        """获取当前上下文"""
        return self._context

    def reset_context(self) -> None:
        """重置上下文"""
        self._context.clear()
