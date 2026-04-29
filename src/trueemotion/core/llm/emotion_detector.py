"""
LLM 情感检测器 v1.14
====================
使用 LLM 进行深度语义情感检测
"""

import logging
from typing import Dict, Optional, Any, List, Tuple

from trueemotion.core.llm.base import BaseLLMClient, LLMError, LLMResponse

logger = logging.getLogger(__name__)


class LLMEmotionDetector:
    """
    LLM 驱动的情感检测器

    使用 LLM 进行语义理解，比规则引擎更准确地检测复杂情感
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        cache_ttl: int = 3600,
    ):
        """
        初始化 LLM 情感检测器

        Args:
            llm_client: LLM 客户端
            cache_ttl: 缓存 TTL（秒），默认 1 小时
        """
        self._llm = llm_client
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = cache_ttl

    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        检测情感 - 返回 dict[str, float] 格式兼容原接口

        Args:
            text: 输入文本
            context: 可选上下文信息

        Returns:
            Dict[str, float]: 情感及其强度分数
        """
        # 检查缓存
        cache_key = text
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit for: {text[:30]}...")
            return cached

        try:
            result = self._llm.detect_emotion(text, context)

            # 提取 all_emotions 作为返回格式
            emotions = result.get("all_emotions", {})

            # 添加复合情感
            for compound in result.get("compound_emotions", []):
                emotions[compound["name"]] = compound["intensity"]

            # 缓存结果
            self._cache[cache_key] = emotions

            return emotions

        except LLMError as e:
            logger.error(f"LLM emotion detection failed: {e}")
            raise

    def get_top_emotions(
        self, text: str, top_k: int = 5, context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str]]:
        """
        获取 Top-K 情感

        Args:
            text: 输入文本
            top_k: 返回数量
            context: 可选上下文

        Returns:
            List[Tuple[emotion, score, explanation]]: Top-K 情感列表
        """
        emotions = self.detect(text, context)

        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        result = []
        for emotion, score in sorted_emotions[:top_k]:
            result.append((emotion, score, ""))

        return result

    def get_detailed_result(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        获取详细检测结果

        Args:
            text: 输入文本
            context: 可选上下文

        Returns:
            Dict: 包含所有检测信息的完整结果
        """
        cache_key = f"_detailed_{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._llm.detect_emotion(text, context)

        # 缓存
        self._cache[cache_key] = result

        return result

    def explain(self, text: str) -> Dict[str, Any]:
        """
        获取情感检测解释

        Args:
            text: 输入文本

        Returns:
            Dict: 包含 explanation 的详细结果
        """
        detailed = self.get_detailed_result(text)
        return {
            "text": text,
            "primary_emotion": detailed.get("primary_emotion"),
            "intensity": detailed.get("intensity"),
            "explanation": detailed.get("explanation", ""),
            "confidence": detailed.get("confidence", 0.0),
            "all_emotions": detailed.get("all_emotions", {}),
            "compound_emotions": detailed.get("compound_emotions", []),
            "vad": detailed.get("vad", {}),
        }

    def clear_cache(self) -> None:
        """清除缓存"""
        self._cache.clear()
        logger.debug("Emotion detection cache cleared")
