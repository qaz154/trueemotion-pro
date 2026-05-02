"""
TrueEmotion Pro v1.15 - 人性化情感AI系统
让AI拥有像人类一样丰富、复杂、真实的情感

v1.15 新特性:
- 修复所有已知严重Bug
- 进化系统真正生效
- 内存系统线程安全与原子写入
- 中文关键词提取优化（jieba支持）
- 响应引擎模板变量与去重
- LLM缓存TTL与可用性检查优化
- 版本统一管理
"""

__version__ = "1.15"
__author__ = "TrueEmotion Team"

from trueemotion.api.routes import TrueEmotionPro, create_analyzer
from trueemotion.api.schemas import (
    AnalyzeRequest,
    EmotionResult,
    EmotionData,
    ResponseData,
    ProfileData,
    EvolutionResult,
    SystemStats,
)
from trueemotion.core.emotions.detector import HumanEmotionDetector

__all__ = [
    # Main API
    "TrueEmotionPro",
    "create_analyzer",
    # Schemas
    "AnalyzeRequest",
    "EmotionResult",
    "EmotionData",
    "ResponseData",
    "ProfileData",
    "EvolutionResult",
    "SystemStats",
    # Core
    "HumanEmotionDetector",
]
