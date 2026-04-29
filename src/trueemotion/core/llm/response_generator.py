"""
LLM 响应生成器 v1.14
====================
使用 LLM 生成自然、口语化的共情响应
"""

import logging
from typing import Optional, Dict, Any, List

from trueemotion.core.llm.base import BaseLLMClient, LLMError
from trueemotion.core.response.engine import EmpathyResponse

logger = logging.getLogger(__name__)


# 情感类型映射
EMPATHY_TYPE_MAPPING = {
    "joy": "分享喜悦",
    "sadness": "深度共情",
    "anger": "安抚情绪",
    "fear": "安全感提供",
    "anxiety": "缓解焦虑",
    "surprise": "好奇回应",
    "love": "温暖回应",
    "trust": "信任回应",
    "anticipation": "期待共鸣",
    "relief": "释放共情",
    "frustration": "理解挫折",
    "disappointment": "共情失落",
    "disgust": "理解接纳",
    "regret": "安慰释怀",
    "guilt": "宽慰自责",
    "pride": "真诚赞美",
    "envy": "理解羡慕",
    "nostalgia": "怀旧共鸣",
    "loneliness": "陪伴温暖",
    "melancholy": "倾听陪伴",
    "bittersweet": "理解复杂",
    "painful_joy": "共情喜悦",
    "hope_fear": "缓解忐忑",
    "jealous_love": "理解吃醋",
    "despair": "陪伴支持",
}


def _get_intensity_level(intensity: float) -> str:
    """获取强度等级"""
    if intensity >= 0.85:
        return "极致"
    elif intensity >= 0.6:
        return "强烈"
    elif intensity >= 0.4:
        return "中等"
    elif intensity >= 0.2:
        return "轻微"
    else:
        return "极微"


def _get_empathy_type(emotion: str) -> str:
    """获取共情类型"""
    return EMPATHY_TYPE_MAPPING.get(emotion, "共情回应")


class LLMResponseGenerator:
    """
    LLM 驱动的响应生成器

    使用 LLM 生成个性化、口语化的共情响应
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        fallback_engine: Optional[Any] = None,
    ):
        """
        初始化 LLM 响应生成器

        Args:
            llm_client: LLM 客户端
            fallback_engine: 可选的降级引擎（当 LLM 不可用时使用）
        """
        self._llm = llm_client
        self._fallback = fallback_engine

    def generate(
        self,
        text: str,
        emotion: str,
        intensity: float,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> EmpathyResponse:
        """
        生成共情响应

        Args:
            text: 用户原始输入
            emotion: 主要情感
            intensity: 情感强度 0.0-1.0
            context: 可选上下文
            user_profile: 用户画像
            conversation_history: 对话历史

        Returns:
            EmpathyResponse: 生成的响应
        """
        try:
            # 使用 LLM 生成响应
            response_text = self._llm.generate_response(
                text=text,
                emotion=emotion,
                intensity=intensity,
                user_profile=user_profile,
                conversation_history=conversation_history,
            )

            # 生成追问
            follow_up = self._maybe_generate_follow_up(text, emotion, intensity)

            return EmpathyResponse(
                text=response_text,
                empathy_type=_get_empathy_type(emotion),
                intensity_level=_get_intensity_level(intensity),
                follow_up=follow_up,
                tone="温暖",
            )

        except LLMError as e:
            logger.warning(f"LLM response generation failed: {e}")
            if self._fallback:
                logger.info("Falling back to template engine")
                return self._fallback.generate(emotion, intensity)
            raise

    def _maybe_generate_follow_up(
        self,
        text: str,
        emotion: str,
        intensity: float,
    ) -> Optional[str]:
        """
        生成追问

        Args:
            text: 用户输入
            emotion: 情感
            intensity: 强度

        Returns:
            Optional[str]: 追问内容
        """
        # 低强度不追问
        if intensity < 0.4:
            return None

        # 高强度大概率追问
        if intensity > 0.7:
            prob = 0.7
        else:
            prob = 0.3

        import random

        if random.random() > prob:
            return None

        follow_up_prompts = {
            "joy": "然后呢？",
            "sadness": "发生什么了？",
            "anger": "怎么了？",
            "fear": "在担心什么？",
            "anxiety": "有什么心事吗？",
            "surprise": "什么情况？！",
            "love": "好甜啊！",
            "trust": "谢谢你信任我",
            "anticipation": "好期待啊！",
            "disappointment": "怎么这样啊...",
            "frustration": "确实挺让人郁闷的",
        }

        return follow_up_prompts.get(emotion, "然后呢？")
