"""
降级管理器 v1.15
================
LLM 故障时自动降级到规则引擎
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime, timedelta

from trueemotion.core.llm.base import LLMError

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    is_half_open: bool = False


class FallbackManager:
    """
    降级管理器

    实现熔断器模式，当 LLM 连续失败时自动降级到规则引擎
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        """
        初始化降级管理器

        Args:
            failure_threshold: 连续失败次数阈值，达到后打开熔断器
            recovery_timeout: 恢复超时时间（秒），熔断打开后等待这么久尝试半开
            half_open_max_calls: 半开状态下允许的最大调用次数
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState()
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        """熔断器是否打开"""
        if not self._state.is_open:
            return False

        # 检查是否应该尝试恢复
        if self._state.last_failure_time:
            elapsed = (datetime.now() - self._state.last_failure_time).total_seconds()
            if elapsed >= self._recovery_timeout:
                self._state.is_open = False
                self._state.is_half_open = True
                self._half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
                return False

        return True

    @property
    def should_fallback(self) -> bool:
        """是否应该降级到规则引擎"""
        return self.is_open

    def record_success(self) -> None:
        """记录成功调用"""
        if self._state.is_half_open:
            self._half_open_calls += 1
            if self._half_open_calls >= self._half_open_max_calls:
                # 连续成功，关闭熔断器
                self._reset()
                logger.info("Circuit breaker closed after successful recovery")
        else:
            # 正常状态，清除失败计数
            self._state.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        记录失败调用

        Args:
            error: 失败原因
        """
        self._state.failure_count += 1
        self._state.last_failure_time = datetime.now()

        if self._state.is_half_open:
            # 半开状态下失败，重新打开熔断器
            self._state.is_open = True
            self._state.is_half_open = False
            logger.warning(f"Circuit breaker re-opened after failure in half-open state: {error}")
        elif self._state.failure_count >= self._failure_threshold:
            # 达到阈值，打开熔断器
            self._state.is_open = True
            logger.warning(f"Circuit breaker opened after {self._failure_threshold} consecutive failures: {error}")

    def _reset(self) -> None:
        """重置熔断器状态"""
        self._state = CircuitBreakerState()
        self._half_open_calls = 0

    def get_status(self) -> dict:
        """获取熔断器状态"""
        return {
            "is_open": self._state.is_open,
            "is_half_open": self._state.is_half_open,
            "failure_count": self._state.failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_seconds": self._recovery_timeout,
        }


class FallbackHandler:
    """
    降级处理器

    包装 LLM 调用，自动处理降级
    """

    def __init__(
        self,
        llm_callable: Callable,
        fallback_callable: Callable,
        fallback_manager: Optional[FallbackManager] = None,
    ):
        """
        初始化降级处理器

        Args:
            llm_callable: LLM 调用函数
            fallback_callable: 降级时的回退函数
            fallback_manager: 降级管理器
        """
        self._llm = llm_callable
        self._fallback = fallback_callable
        self._manager = fallback_manager or FallbackManager()

    def call(self, *args, **kwargs) -> Any:
        """
        执行调用，失败时自动降级

        Returns:
            LLM 结果或降级结果
        """
        if self._manager.should_fallback:
            logger.debug("Circuit breaker is open, using fallback")
            return self._fallback(*args, **kwargs)

        try:
            result = self._llm(*args, **kwargs)
            self._manager.record_success()
            return result
        except LLMError as e:
            self._manager.record_failure(e)
            logger.warning(f"LLM call failed, falling back: {e}")
            return self._fallback(*args, **kwargs)

    def get_status(self) -> dict:
        """获取降级状态"""
        return {
            "circuit_breaker": self._manager.get_status(),
        }
