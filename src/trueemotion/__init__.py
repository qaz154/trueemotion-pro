"""
TrueEmotion Pro v1.16 - 人性化情感AI系统
让AI拥有像人类一样丰富、复杂、真实的情感

v1.16 新特性:
- OpenAI 客户端改用官方 SDK（替换 urllib.request）
- 清理仓库：移除提交的模型文件和临时文件
- pyproject.toml 声明实际依赖
- 版本统一管理
"""

__version__ = "1.16"
__author__ = "TrueEmotion Team"

from trueemotion.api.routes import TrueEmotionPro, create_analyzer
from trueemotion.core.emotions.detector import HumanEmotionDetector

__all__ = [
    "TrueEmotionPro",
    "create_analyzer",
    "HumanEmotionDetector",
]
