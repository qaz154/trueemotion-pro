"""
性格与关系系统 v1.11
====================
让AI拥有独特的性格和关系感知能力

性格维度:
1. 外向性 - 外放 vs 内敛
2. 敏感度 - 敏感 vs 迟钝
3. 表达方式 - 直接 vs 含蓄
4. 共情深度 - 深度共情 vs 理性分析
5. 情绪稳定性 - 波动 vs 平稳
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import random


class PersonalityTrait(Enum):
    """性格特征"""
    EXTROVERT = "extrovert"       # 外向
    INTROVERT = "introvert"      # 内敛
    SENSITIVE = "sensitive"       # 敏感
    RESILIENT = "resilient"      # 坚强
    DIRECT = "direct"             # 直接
    INDIRECT = "indirect"        # 含蓄
    DEEP_EMPATH = "deep_empath"  # 深度共情
    ANALYTICAL = "analytical"    # 理性分析
    EXPRESSIVE = "expressive"     # 善于表达
    RESTRAINED = "restrained"    # 克制内敛


@dataclass
class Personality:
    """
    性格模型

    每个维度0.0-1.0，0.5为中性
    """
    # 外向性 (0=极度内向, 1=极度外向)
    extroversion: float = 0.5

    # 敏感度 (0=迟钝, 1=高度敏感)
    sensitivity: float = 0.5

    # 表达直接性 (0=含蓄, 1=直接)
    directness: float = 0.5

    # 共情倾向 (0=理性, 1=共情)
    empathy_tendency: float = 0.6

    # 情绪稳定性 (0=波动, 1=稳定)
    emotional_stability: float = 0.6

    # 幽默感 (0=严肃, 1=幽默)
    humor: float = 0.4

    # 温暖度 (0=冷淡, 1=温暖)
    warmth: float = 0.7

    def get_trait_description(self, trait: PersonalityTrait) -> str:
        """获取性格特征描述"""
        descriptions = {
            PersonalityTrait.EXTROVERT: "热情开朗，善于社交",
            PersonalityTrait.INTROVERT: "沉静内敛，深思熟虑",
            PersonalityTrait.SENSITIVE: "细腻敏感，体察入微",
            PersonalityTrait.RESILIENT: "坚强乐观，抗压能力强",
            PersonalityTrait.DIRECT: "直接坦诚，有话直说",
            PersonalityTrait.INDIRECT: "委婉含蓄，说话有分寸",
            PersonalityTrait.DEEP_EMPATH: "善解人意，感同身受",
            PersonalityTrait.ANALYTICAL: "理性客观，善于分析",
            PersonalityTrait.EXPRESSIVE: "表达丰富，善于沟通",
            PersonalityTrait.RESTRAINED: "克制内敛，话少但精",
        }
        return descriptions.get(trait, "")

    def get_response_style(self) -> str:
        """获取响应风格描述"""
        styles = []
        if self.extroversion > 0.7:
            styles.append("热情")
        elif self.extroversion < 0.3:
            styles.append("内敛")

        if self.humor > 0.6:
            styles.append("幽默")
        elif self.humor < 0.3:
            styles.append("认真")

        if self.warmth > 0.7:
            styles.append("温暖")
        elif self.warmth < 0.4:
            styles.append("冷静")

        if self.directness > 0.7:
            styles.append("直接")

        return "的".join(styles) if styles else "平和"


@dataclass
class Relationship:
    """
    关系模型

    记录与用户的互动历史和关系亲密度
    """
    user_id: str
    familiarity: float = 0.3       # 熟悉度 0-1
    trust_level: float = 0.3       # 信任度 0-1
    emotional_bond: float = 0.3    # 情感纽带 0-1
    interaction_count: int = 0     # 互动次数
    positive_ratio: float = 0.5    # 正面互动比例
    topics_discussed: List[str] = field(default_factory=list)  # 讨论过的话题
    last_interaction: Optional[str] = None

    def get_intimacy_level(self) -> str:
        """获取亲密度等级"""
        avg = (self.familiarity + self.trust_level + self.emotional_bond) / 3
        if avg >= 0.8:
            return "亲密"
        elif avg >= 0.6:
            return "熟悉"
        elif avg >= 0.4:
            return "一般"
        elif avg >= 0.2:
            return "陌生"
        return "初识"

    def should_use_informal_tone(self) -> bool:
        """是否使用非正式语气（亲密时用"你"，生疏时用"您"）"""
        return self.familiarity > 0.5 and self.trust_level > 0.4

    def adjust_response_for_relationship(self, base_response: str) -> str:
        """根据关系调整回复"""
        if not self.should_use_informal_tone():
            # 生疏时用更礼貌的称呼
            if base_response.startswith("你"):
                return "您" + base_response[1:]
        return base_response


class PersonalityEngine:
    """
    性格引擎

    根据性格和关系生成个性化的响应
    """

    # 默认性格配置
    DEFAULT_PERSONALITY = Personality(
        extroversion=0.5,
        sensitivity=0.6,
        directness=0.4,
        empathy_tendency=0.7,
        emotional_stability=0.6,
        humor=0.5,
        warmth=0.7,
    )

    def __init__(self, personality: Optional[Personality] = None):
        self.personality = personality or self.DEFAULT_PERSONALITY

    def get_response_modifier(
        self,
        emotion: str,
        intensity: float,
        relationship: Optional[Relationship] = None,
    ) -> Dict:
        """
        根据性格和关系获取响应修饰符

        Returns:
            Dict: 包含 tone, formality, warmth 等修饰符
        """
        modifiers = {
            "tone": self._get_tone(emotion, intensity),
            "formality": self._get_formality(relationship),
            "warmth": self._get_warmth_modifier(),
            "humor_appropriateness": self._get_humor_appropriateness(emotion, intensity),
            "directness": self._get_directness_modifier(emotion),
            "empathy_depth": self._get_empathy_depth(),
        }

        return modifiers

    def _get_tone(self, emotion: str, intensity: float) -> str:
        """获取语气"""
        if intensity > 0.85:
            # 强烈情感 - 根据性格调整
            if self.personality.extroversion > 0.6:
                return "激动"
            else:
                return "深沉"
        elif intensity > 0.5:
            return "温和"
        else:
            return "平静"

    def _get_formality(self, relationship: Optional[Relationship]) -> str:
        """获取正式度"""
        if relationship and relationship.should_use_informal_tone():
            return "非正式"
        return "适度正式"

    def _get_warmth_modifier(self) -> float:
        """获取温暖度修饰符"""
        return self.personality.warmth

    def _get_humor_appropriateness(self, emotion: str, intensity: float) -> float:
        """获取幽默适当度"""
        # 强烈负面情绪时不适用幽默
        if emotion in ["sadness", "grief", "despair", "fear", "terror"]:
            return 0.0

        # 愤怒时少量幽默可以缓解
        if emotion in ["anger", "rage", "annoyance"]:
            return self.personality.humor * 0.3

        # 正面情绪时可以使用幽默
        if emotion in ["joy", "ecstasy", "amusement", "surprise"]:
            return self.personality.humor * 0.8

        return self.personality.humor * 0.5

    def _get_directness_modifier(self, emotion: str) -> float:
        """获取直接度修饰符"""
        base = self.personality.directness

        # 悲伤时减少直接度，增加共情
        if emotion in ["sadness", "grief", "despair"]:
            return base * 0.7

        # 愤怒时根据性格决定是否直接
        if emotion in ["anger", "rage"]:
            return base * 0.8

        return base

    def _get_empathy_depth(self) -> str:
        """获取共情深度"""
        if self.personality.empathy_tendency > 0.75:
            return "深度共情"
        elif self.personality.empathy_tendency > 0.5:
            return "适度共情"
        elif self.personality.empathy_tendency > 0.25:
            return "轻度共情"
        return "理性回应"

    def adapt_response(
        self,
        base_response: str,
        emotion: str,
        intensity: float,
        relationship: Optional[Relationship] = None,
    ) -> str:
        """
        根据性格和关系调整回复

        Args:
            base_response: 基础回复
            emotion: 当前情感
            intensity: 情感强度
            relationship: 用户关系

        Returns:
            str: 调整后的回复
        """
        response = base_response

        # 根据关系调整称呼
        if relationship:
            response = relationship.adjust_response_for_relationship(response)

        # 根据性格调整表达
        if self.personality.warmth > 0.7 and intensity > 0.5:
            # 高温暖性格在强烈情感时增加表达
            if not any(marker in response for marker in ["！", "啊", "呀", "哇"]):
                if emotion in ["joy", "ecstasy", "surprise"]:
                    response = response.rstrip("。") + "！"

        # 内敛性格适当收敛表达
        if self.personality.extroversion < 0.4:
            if "！！" in response or "！！" in response:
                response = response.replace("！！", "！")

        return response
