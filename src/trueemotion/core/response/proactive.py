"""
主动共情系统 v1.15
====================
让AI不只能回应，还能主动关心

核心理念:
1. 主动关心 - 不只是回答，而是主动问候
2. 情感预测 - 根据上下文预测可能需要关心
3. 自然发起 - 像真人一样自然的主动关怀
4. 适可而止 - 不会过于频繁或打扰
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import random


@dataclass
class ProactiveResponse:
    """主动响应"""
    should_initiate: bool
    message: Optional[str]
    reason: str  # 为什么发起主动关怀
    priority: str  # "high", "medium", "low"


class ProactiveEmpathyEngine:
    """
    主动共情引擎

    在合适的时机主动发起关怀，让AI更像真人
    """

    # 主动关怀触发条件
    CARE_TRIGGERS = {
        "long_absense": {
            "threshold_hours": 24,
            "message": "嗨，好久不见，你还好吗？",
            "priority": "medium",
        },
        "negative_trend": {
            "threshold": 2,  # 连续N轮负面
            "emotions": ["sadness", "fear", "anxiety", "despair"],
            "priority": "high",
        },
        "time_based": {
            "evening": {
                "hours": (20, 23),
                "message": "这么晚了，还好吗？注意休息啊",
                "priority": "low",
            },
            "late_night": {
                "hours": (23, 6),
                "message": "这么晚还没睡？注意身体啊",
                "priority": "medium",
            },
        },
    }

    # 持续负面情感的关怀
    SUSTAINED_CARE = {
        "sadness": [
            "在想那件事吗？还好吗？",
            "我有点担心你，还好吗？",
            "有什么想说的吗？我在",
        ],
        "fear": [
            "还在担心吗？我在听着",
            "能说说是什么让你害怕吗？",
        ],
        "anxiety": [
            "事情解决了吗？需要帮忙吗？",
            "别着急，我陪你一起想想",
        ],
        "despair": [
            "希望还在吗？我们一起想办法",
            "不管怎样，我都在这里",
        ],
    }

    # 积极情感的共鸣
    POSITIVE_SHARE = {
        "joy": [
            "有什么好事发生了吗？想分享吗？",
            "看起来心情不错，发生什么了？",
        ],
        "ecstasy": [
            "太开心了吧！什么事让你这么高兴？",
            "哇，看起来真的很开心！说说看！",
        ],
        "pride": [
            "有什么值得骄傲的事吗？",
            "听起来你很有成就感！",
        ],
    }

    def __init__(self):
        self._last_proactive_time: Optional[datetime] = None
        self._proactive_cooldown_hours = 2  # 至少间隔2小时
        self._negative_count = 0

    def should_initiate(
        self,
        recent_emotions: List[str],
        time_since_last: Optional[datetime],
        current_hour: int,
        current_emotion: Optional[str] = None,
        current_intensity: float = 0.0,
    ) -> ProactiveResponse:
        """
        判断是否应该主动发起关怀

        Args:
            recent_emotions: 最近N轮的情感
            time_since_last: 距离上次互动的时间
            current_hour: 当前小时
            current_emotion: 当前情感
            current_intensity: 当前强度

        Returns:
            ProactiveResponse: 是否主动响应
        """
        # 1. 检查冷却期
        hours_passed = None
        if time_since_last:
            hours_passed = (datetime.now() - time_since_last).total_seconds() / 3600
            if hours_passed < self._proactive_cooldown_hours:
                return ProactiveResponse(False, None, "冷却期内", "low")

        # 2. 检查长期未互动
        if hours_passed is not None:
            if hours_passed >= self.CARE_TRIGGERS["long_absense"]["threshold_hours"]:
                return ProactiveResponse(
                    True,
                    self.CARE_TRIGGERS["long_absense"]["message"],
                    f"超过{hours_passed:.0f}小时未互动",
                    "medium"
                )

        # 3. 检查负面情感趋势
        negative_emotions = set(self.CARE_TRIGGERS["negative_trend"]["emotions"])
        recent_negative = sum(1 for e in recent_emotions if e in negative_emotions)

        if current_emotion in negative_emotions and current_intensity > 0.3:
            self._negative_count += 1
        else:
            self._negative_count = 0

        if self._negative_count >= self.CARE_TRIGGERS["negative_trend"]["threshold"]:
            message = self._get_sustained_care_message(current_emotion)
            return ProactiveResponse(
                True,
                message,
                "持续负面情感",
                "high"
            )

        # 4. 检查时间相关关怀（需检查主动消息冷却期）
        if self._last_proactive_time:
            hours_since_proactive = (datetime.now() - self._last_proactive_time).total_seconds() / 3600
            if hours_since_proactive < self._proactive_cooldown_hours:
                return ProactiveResponse(False, None, "主动消息冷却期内", "low")

        if 20 <= current_hour < 23:
            self.record_proactive()
            return ProactiveResponse(
                True,
                self.CARE_TRIGGERS["time_based"]["evening"]["message"],
                "晚间问候",
                "low"
            )
        elif 23 <= current_hour or current_hour < 6:
            self.record_proactive()
            return ProactiveResponse(
                True,
                self.CARE_TRIGGERS["time_based"]["late_night"]["message"],
                "深夜关怀",
                "medium"
            )

        # 5. 检查积极情感共鸣
        if current_emotion in self.POSITIVE_SHARE:
            if current_intensity > 0.6 and random.random() < 0.3:
                message = random.choice(self.POSITIVE_SHARE[current_emotion])
                return ProactiveResponse(
                    True,
                    message,
                    "积极情感共鸣",
                    "low"
                )

        return ProactiveResponse(False, None, "无需主动", "low")

    def _get_sustained_care_message(self, emotion: Optional[str]) -> str:
        """获取持续负面情感的关怀消息"""
        if emotion and emotion in self.SUSTAINED_CARE:
            return random.choice(self.SUSTAINED_CARE[emotion])

        # 默认关怀
        default_messages = [
            "你还好吗？我有点担心你",
            "有什么想说的吗？我在听着",
            "需要帮忙吗？我陪你一起想想",
        ]
        return random.choice(default_messages)

    def record_proactive(self) -> None:
        """记录主动发起的关怀"""
        self._last_proactive_time = datetime.now()

    def reset_counters(self) -> None:
        """重置计数器"""
        self._negative_count = 0


class ResponseDiversity:
    """
    响应多样性

    让同样的情感表达有多种不同的回复方式
    """

    # 不同风格的响应
    RESPONSE_STYLES = {
        "warm": [  # 温暖风格
            "我在这里呢",
            "我一直陪着你",
            "有我在",
            "别担心，我在",
        ],
        "casual": [  # 轻松风格
            "哎，怎么了？",
            "嗨，怎么啦？",
            "嘿，我听着呢",
            "在呢，怎么了？",
        ],
        "gentle": [  # 温柔风格
            "慢慢说，我听着",
            "不着急，我在这里",
            "想说什么就说",
            "我陪着你呢",
        ],
        "direct": [  # 直接风格
            "说吧，怎么了？",
            "什么事？",
            "我在听",
            "怎么了？",
        ],
    }

    # 追问多样性
    FOLLOW_UP_QUESTIONS = {
        "joy": [
            "然后呢然后呢？",
            "详细讲讲呗",
            "快说说！",
            "太棒了，怎么做到的？",
            "让你这么开心的是什么呀？",
        ],
        "sadness": [
            "想说就说，我陪你",
            "发生什么了？",
            "怎么了吗？",
            "我听着呢",
            "想聊聊吗？",
        ],
        "anger": [
            "什么事让你这么气？",
            "说说看，我帮你分析",
            "怎么了？",
            "谁惹你了？",
        ],
        "fear": [
            "在担心什么？",
            "能说说吗？",
            "我陪着你，别怕",
            "怎么了？",
        ],
        "neutral": [
            "然后呢？",
            "嗯嗯，继续说",
            "嗯，我在听",
            "怎么想的？",
        ],
    }

    @classmethod
    def get_varied_response(
        cls,
        emotion: str,
        intensity: float,
        style: Optional[str] = None,
    ) -> str:
        """
        获取多样化的响应

        Args:
            emotion: 情感类型
            intensity: 强度
            style: 可选的风格偏好

        Returns:
            str: 多样化的响应
        """
        # 确定风格
        if style is None:
            if intensity > 0.7:
                style = "warm"
            elif intensity > 0.4:
                style = "gentle"
            else:
                style = "casual"

        # 随机选择
        responses = cls.RESPONSE_STYLES.get(style, cls.RESPONSE_STYLES["warm"])
        return random.choice(responses)

    @classmethod
    def get_varied_follow_up(cls, emotion: str) -> str:
        """获取多样化的追问"""
        questions = cls.FOLLOW_UP_QUESTIONS.get(emotion, cls.FOLLOW_UP_QUESTIONS["neutral"])
        return random.choice(questions)
