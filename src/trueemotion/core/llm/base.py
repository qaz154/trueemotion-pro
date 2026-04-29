"""
LLM 客户端抽象接口 v1.14
========================
定义 LLM 交互的统一接口，支持多种 Provider
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMError(Exception):
    """LLM 相关错误"""
    pass


class BaseLLMClient(ABC):
    """
    LLM 客户端抽象接口

    所有 LLM Provider 需要实现此接口
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本补全

        Args:
            prompt: 输入提示词
            temperature: 温度参数 (0.0-1.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他 provider 相关参数

        Returns:
            LLMResponse: LLM 响应
        """
        pass

    @abstractmethod
    def detect_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        使用 LLM 检测情感

        Args:
            text: 输入文本
            context: 可选上下文信息

        Returns:
            Dict: 包含 primary_emotion, intensity, all_emotions, vad, explanation, confidence
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        text: str,
        emotion: str,
        intensity: float,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> str:
        """
        使用 LLM 生成共情响应

        Args:
            text: 用户原始输入
            emotion: 检测到的情感
            intensity: 情感强度 0.0-1.0
            user_profile: 用户画像
            conversation_history: 对话历史

        Returns:
            str: 生成的共情回复
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查 LLM 服务是否可用

        Returns:
            bool: 是否可用
        """
        pass

    def _record_latency(self, start_time: float) -> float:
        """记录延迟"""
        return (time.time() - start_time) * 1000
