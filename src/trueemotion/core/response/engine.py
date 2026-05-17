"""
人性化共情响应引擎
=========================
让AI的回复像真人一样自然、有温度

核心理念:
1. 真实感 - 不像模板，有随机性
2. 共情深度 - 不是简单安慰，是真正的理解
3. 个性化 - 根据性格和关系调整
4. 细腻度 - 考虑情感复合、强度变化
5. 口语化 - 符合日常对话习惯
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
import random

from trueemotion.core.emotions.personality import Personality, Relationship, PersonalityEngine
from trueemotion.core.emotions.i18n import EMOTION_CN
from trueemotion.core.response.templates import EMPATHETIC_RESPONSES, FOLLOW_UP_TEMPLATES, FILLERS


@dataclass
class EmpathyResponse:
    """共情响应"""
    text: str
    empathy_type: str           # support, comfort, excitement, calm, etc.
    intensity_level: str        # 极致, 强烈, 中等, 轻微, 极微
    follow_up: Optional[str] = None
    tone: str = "温暖"         # 语气描述
    adaptation_notes: List[str] = None  # 调整说明

    def __post_init__(self):
        if self.adaptation_notes is None:
            self.adaptation_notes = []


class HumanEmpathyEngine:
    """
    人性化共情引擎

    特点:
    1. 多层响应模板 - 覆盖不同情感和强度
    2. 随机性 - 同样的输入不总是同样的输出
    3. 复合情感支持 - 如"悲喜交加"有特殊响应
    4. 性格适应 - 根据配置的性格调整响应
    5. 关系感知 - 根据亲密度调整语气
    """

    # 强度等级阈值
    INTENSITY_HIGH_THRESHOLD = 0.85
    INTENSITY_MEDIUM_THRESHOLD = 0.50
    INTENSITY_LOW_THRESHOLD = 0.20

    # 随机性概率常量
    EMPHASIS_PROBABILITY = 0.3
    FOLLOW_UP_HIGH_PROBABILITY = 0.7
    FOLLOW_UP_MEDIUM_PROBABILITY = 0.4
    FILLER_PROBABILITY = 0.25
    EMPHASIS_INTENSITY_THRESHOLD = 0.8
    FILLER_INTENSITY_THRESHOLD = 0.6

    # 强度标签
    INTENSITY_LEVEL_HIGH = "high"
    INTENSITY_LEVEL_MEDIUM = "medium"
    INTENSITY_LEVEL_LOW = "low"
    INTENSITY_LEVEL_MINIMAL = "minimal"

    def __init__(
        self,
        personality: Optional[Personality] = None,
        personality_engine: Optional[PersonalityEngine] = None,
    ):
        self._personality = personality or Personality()
        self._personality_engine = personality_engine or PersonalityEngine(self._personality)
        self._recent_responses: deque = deque(maxlen=20)

    # ============================================================
    # 核心方法
    # ============================================================

    def _substitute_variables(self, response: str, text: str, emotion: str) -> str:
        """模板变量替换"""
        response = response.replace("{emotion_word}", EMOTION_CN.get(emotion, ""))
        response = response.replace("{text_summary}", text[:10] if len(text) > 10 else text)
        return response

    def generate(
        self,
        emotion: str,
        intensity: float,
        context: Optional[str] = None,
        relationship: Optional[Relationship] = None,
    ) -> EmpathyResponse:
        """
        生成共情响应

        Args:
            emotion: 主要情感
            intensity: 强度 0.0-1.0
            context: 可选上下文
            relationship: 可选关系信息

        Returns:
            EmpathyResponse: 生成的响应
        """
        # 1. 确定强度等级
        intensity_level = self._get_intensity_level(intensity)

        # 2. 获取基础响应
        response_text = self._get_base_response(emotion, intensity_level)

        # 2.5 模板变量替换
        response_text = self._substitute_variables(response_text, context or "", emotion)

        # 3. 添加随机性
        response_text = self._add_randomness(response_text, emotion, intensity)

        # 4. 可能添加追问
        follow_up = self._maybe_add_follow_up(emotion, intensity_level)

        # 5. 添加语气词
        response_text = self._add_filler(response_text, intensity)

        # 6. 根据性格和关系调整
        response_text = self._personality_engine.adapt_response(
            response_text, emotion, intensity, relationship
        )

        # 7. 获取响应类型
        empathy_type = self._get_empathy_type(emotion, intensity)

        return EmpathyResponse(
            text=response_text,
            empathy_type=empathy_type,
            intensity_level=intensity_level,
            follow_up=follow_up,
            tone=self._personality_engine._get_tone(emotion, intensity),
        )

    def _get_intensity_level(self, intensity: float) -> str:
        """获取强度等级

        调整阈值以更好匹配实际情感强度:
        - 强烈负面情感(如绝望)即使分数不高也应有更深入的回应
        """
        if intensity >= self.INTENSITY_HIGH_THRESHOLD:
            return self.INTENSITY_LEVEL_HIGH
        elif intensity >= self.INTENSITY_MEDIUM_THRESHOLD:
            return self.INTENSITY_LEVEL_MEDIUM
        elif intensity >= self.INTENSITY_LOW_THRESHOLD:
            return self.INTENSITY_LEVEL_LOW
        else:
            return self.INTENSITY_LEVEL_MINIMAL

    def _get_base_response(self, emotion: str, intensity_level: str) -> str:
        """获取基础响应"""
        # 尝试从对应情感获取
        if emotion in EMPATHETIC_RESPONSES:
            templates = (
                EMPATHETIC_RESPONSES[emotion].get(intensity_level) or
                EMPATHETIC_RESPONSES[emotion].get("low") or
                EMPATHETIC_RESPONSES[emotion].get("minimal") or
                ["嗯"]
            )
        else:
            # 回退到默认
            templates = (
                EMPATHETIC_RESPONSES.get("default", {}).get(intensity_level) or
                EMPATHETIC_RESPONSES.get("default", {}).get("low") or
                ["嗯"]
            )

        # Filter out recently used responses
        available = [t for t in templates if t not in self._recent_responses]
        if not available:
            available = templates  # All used, reset
        choice = random.choice(available)
        self._recent_responses.append(choice)
        return choice

    def _add_randomness(
        self,
        response: str,
        emotion: str,
        intensity: float,
    ) -> str:
        """添加随机性，让同样输入有不同输出"""
        # 高强度情感时偶尔添加强调
        if intensity > self.EMPHASIS_INTENSITY_THRESHOLD and random.random() < self.EMPHASIS_PROBABILITY:
            emphasis = random.choice(["真的", "确实", "完全", ""])
            if emphasis and not response.startswith(emphasis):
                response = emphasis + response

        return response

    def _maybe_add_follow_up(
        self,
        emotion: str,
        intensity_level: str,
    ) -> Optional[str]:
        """可能添加追问"""
        # 中高强度时更可能添加追问
        if intensity_level == self.INTENSITY_LEVEL_HIGH and random.random() < self.FOLLOW_UP_HIGH_PROBABILITY:
            templates = FOLLOW_UP_TEMPLATES.get(
                emotion, FOLLOW_UP_TEMPLATES["default"]
            ).get(intensity_level, FOLLOW_UP_TEMPLATES["default"]["low"])
            return random.choice(templates)

        elif intensity_level == self.INTENSITY_LEVEL_MEDIUM and random.random() < self.FOLLOW_UP_MEDIUM_PROBABILITY:
            templates = FOLLOW_UP_TEMPLATES.get(
                emotion, FOLLOW_UP_TEMPLATES["default"]
            ).get("low", ["嗯"])
            return random.choice(templates)

        return None

    def _add_filler(self, response: str, intensity: float) -> str:
        """添加语气词，让语言更自然"""
        # 高强度时偶尔添加语气词
        if intensity > self.FILLER_INTENSITY_THRESHOLD and random.random() < self.FILLER_PROBABILITY:
            filler = random.choice(FILLERS)
            if filler:
                # 根据句尾标点决定插入位置
                if response.endswith("！"):
                    return response[:-1] + filler + "！"
                elif response.endswith("。"):
                    return response[:-1] + filler + "。"
        return response

    def _get_empathy_type(self, emotion: str, intensity: float) -> str:
        """获取响应类型"""
        type_mapping = {
            "joy": "分享喜悦" if intensity > 0.5 else "温和回应",
            "sadness": "深度共情",
            "anger": "安抚情绪",
            "fear": "安全感提供",
            "anxiety": "缓解焦虑",
            "surprise": "好奇回应",
            "love": "温暖回应",
            "gratitude": "谦逊回应",
            "guilt": "安慰释怀",
            "pride": "真诚赞美",
            "despair": "陪伴支持",
            "confusion": "理清思路",
            "bittersweet": "理解复杂",
            "boredom": "缓解倦怠",
            "loneliness": "陪伴温暖",
            "melancholy": "倾听陪伴",
            "disgust": "理解接纳",
            "trust": "信任回应",
            "anticipation": "期待共鸣",
        }
        return type_mapping.get(emotion, "共情回应")

    def generate_compound_response(
        self,
        emotions: Dict[str, float],
        relationship: Optional[Relationship] = None,
    ) -> EmpathyResponse:
        """
        为复合情感生成响应

        Args:
            emotions: 情感字典 (情感 -> 强度)
            relationship: 关系信息
        """
        if len(emotions) == 1:
            emotion, intensity = list(emotions.items())[0]
            return self.generate(emotion, intensity, relationship=relationship)

        # 复合情感处理
        # 按强度排序
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, primary_intensity = sorted_emotions[0]

        # 检查复合情感
        emotion_keys = set(emotions.keys())

        # 悲喜交加
        if "joy" in emotion_keys and "sadness" in emotion_keys:
            return self.generate("bittersweet", primary_intensity, relationship=relationship)

        # 爱+信任
        if "love" in emotion_keys and "trust" in emotion_keys:
            return self.generate("love", primary_intensity, relationship=relationship)

        # 希望+恐惧
        if "hope" in emotion_keys and "fear" in emotion_keys:
            return self.generate("anxiety", primary_intensity, relationship=relationship)

        # 愤怒+厌恶
        if "anger" in emotion_keys and "disgust" in emotion_keys:
            return self.generate("contempt", primary_intensity, relationship=relationship)

        return self.generate(primary_emotion, primary_intensity, relationship=relationship)
