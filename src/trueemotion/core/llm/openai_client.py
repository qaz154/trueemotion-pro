"""
OpenAI LLM 客户端 v1.16
=======================
基于 OpenAI 官方 SDK 的 LLM 客户端实现

v1.16 变更:
- 使用 openai SDK 替代 urllib.request（自动重试、连接池、更好的错误处理）
- 移除手动重试逻辑（SDK 内置）
"""

import json
import os
import time
from typing import Optional, Dict, Any, List

from trueemotion.core.llm.base import BaseLLMClient, LLMResponse, LLMError
from trueemotion.core.llm.prompts import (
    build_emotion_detection_prompt,
    build_response_generation_prompt,
    EMOTION_DETECTION_FEW_SHOT,
)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API 客户端 v1.16

    使用 openai 官方 SDK，支持 OpenAI 兼容的 API（如 Azure OpenAI、自建代理等）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        初始化 OpenAI 客户端

        Args:
            api_key: OpenAI API Key (默认从环境变量 OPENAI_API_KEY 获取)
            model: 模型名称，默认 gpt-4o-mini
            base_url: API 基础 URL (用于代理或 Azure OpenAI)
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self._timeout = timeout
        self._max_retries = max_retries
        self._last_check_time = None
        self._last_check_result = False

        if not self._api_key:
            raise LLMError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # 初始化 openai SDK 客户端
        try:
            from openai import OpenAI
            client_kwargs = {
                "api_key": self._api_key,
                "timeout": self._timeout,
                "max_retries": self._max_retries,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = OpenAI(**client_kwargs)
        except ImportError:
            raise LLMError(
                "openai SDK not installed. Run: pip install openai>=1.0.0"
            )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> LLMResponse:
        """生成文本补全"""
        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = self._record_latency(start_time)

            choice = response.choices[0] if response.choices else None
            if not choice:
                raise LLMError("No response from OpenAI")

            content = choice.message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model or self._model,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
            )

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"OpenAI request failed: {e}")

    def detect_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        使用 LLM 检测情感

        Args:
            text: 输入文本
            context: 可选上下文信息

        Returns:
            Dict: 包含 primary_emotion, intensity, all_emotions, vad, explanation, confidence
        """
        system_prompt, user_prompt = build_emotion_detection_prompt(text)

        full_prompt = f"{system_prompt}\n{EMOTION_DETECTION_FEW_SHOT}\n{user_prompt}"

        response = self.complete(full_prompt, temperature=0.3, max_tokens=300)

        try:
            result = json.loads(response.content)

            # 验证必要字段
            required = ["primary_emotion", "intensity", "all_emotions"]
            for field in required:
                if field not in result:
                    raise LLMError(f"Missing required field: {field}")

            # 添加原始文本和 LLM 信息
            result["_text"] = text
            result["_model"] = response.model
            result["_latency_ms"] = response.latency_ms

            return result

        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM response as JSON: {e}\nContent: {response.content}")

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
        # 构建上下文信息
        context_parts = []
        if conversation_history:
            recent = conversation_history[-3:]
            context_parts.append(f"对话历史: {' | '.join(recent)}")
        if user_profile:
            context_parts.append(f"用户关系深度: {user_profile.get('relationship_level', 0.5):.1f}")

        context_info = "\n".join(context_parts) if context_parts else ""

        system_prompt, user_prompt = build_response_generation_prompt(
            text=text,
            emotion=emotion,
            intensity=intensity,
            context_info=context_info,
        )

        # 根据情感强度调整 temperature
        temp = 0.9 if intensity > 0.7 else (0.7 if intensity > 0.4 else 0.6)

        response = self.complete(system_prompt + "\n" + user_prompt, temperature=temp, max_tokens=500)

        return response.content.strip()

    def is_available(self) -> bool:
        """检查 LLM 服务是否可用"""
        if self._last_check_time is not None and time.time() - self._last_check_time <= 60:
            return self._last_check_result
        try:
            self.complete("hello", temperature=0.1, max_tokens=5)
            self._last_check_result = True
        except LLMError:
            self._last_check_result = False
        self._last_check_time = time.time()
        return self._last_check_result
