"""
TrueEmotion Pro v1.13 API
人性化情感AI系统
"""

from typing import Optional

from trueemotion.core.analysis.analyzer import EmotionAnalyzer, AnalyzeOptions
from trueemotion.core.analysis.output import AnalysisResult
from trueemotion.learning.evolution import EvolutionManager
from trueemotion.memory.repository import MemoryRepository


class TrueEmotionPro:
    """
    TrueEmotion Pro 主类

    使用方法:
        pro = TrueEmotionPro()
        result = pro.analyze("今天太开心了！")
        print(result.emotion.primary)  # joy
        print(result.human_response.text)  # "太为你高兴了！说说怎么回事！"
    """

    def __init__(
        self,
        memory_path: str = "./memory",
        auto_learn: bool = True,
    ):
        """
        初始化TrueEmotion Pro

        Args:
            memory_path: 记忆存储路径
            auto_learn: 是否自动学习用户反馈
        """
        self._memory = MemoryRepository(memory_path)
        self._analyzer = EmotionAnalyzer(
            memory_path=memory_path,
            detector=None,
            empathy_engine=None,
        )
        self._evolution = EvolutionManager(self._memory)
        self._auto_learn = auto_learn

    def analyze(
        self,
        text: str,
        context: Optional[str] = None,
        learn: bool = False,
        response: Optional[str] = None,
        feedback: float = 0.5,
        user_id: str = "default",
    ) -> AnalysisResult:
        """
        分析文本情感并生成共情回复

        Args:
            text: 输入文本
            context: 对话上下文
            learn: 是否学习此次交互
            response: AI的回复（用于学习）
            feedback: 用户反馈 0-1
            user_id: 用户ID

        Returns:
            AnalysisResult: 完整分析结果
        """
        options = AnalyzeOptions(
            context=context,
            learn=learn or self._auto_learn,
            response=response,
            feedback=feedback,
            user_id=user_id,
        )

        return self._analyzer.analyze(text, options)

    def get_user_profile(self, user_id: str = "default") -> dict:
        """
        获取用户画像

        Args:
            user_id: 用户ID

        Returns:
            dict: 用户画像
        """
        profile = self._memory.get_user(user_id)
        return {
            "user_id": profile.user_id,
            "total_interactions": profile.total_interactions,
            "dominant_emotion": profile.dominant_emotion,
            "relationship_level": profile.relationship_level,
            "learned_patterns": profile.learned_patterns,
            "last_emotion": profile.last_emotion,
            "emotional_history": profile.emotional_history[-10:],  # 最近10条
        }

    def get_memory_status(self) -> dict:
        """
        获取记忆状态

        Returns:
            dict: 记忆系统状态
        """
        return self._memory.get_stats()

    def evolve(self) -> dict:
        """
        执行进化

        分析学习到的模式，提取高反馈的模式作为新规则

        Returns:
            dict: 进化结果
        """
        return self._evolution.evolve()

    def get_evolution_status(self) -> dict:
        """
        获取进化状态

        Returns:
            dict: 进化系统状态
        """
        return self._evolution.get_evolution_status()

    def reset_user(self, user_id: str) -> None:
        """
        重置用户数据

        Args:
            user_id: 用户ID
        """
        self._memory.delete_user(user_id)

    def get_stats(self) -> dict:
        """
        获取系统统计

        Returns:
            dict: 系统统计信息
        """
        stats = self._memory.get_stats()
        stats["version"] = "1.13"
        stats["evolution"] = self._evolution.get_evolution_status()
        return stats


# 便捷函数
def create_analyzer(memory_path: str = "./memory") -> TrueEmotionPro:
    """
    创建TrueEmotion Pro实例

    Args:
        memory_path: 记忆存储路径

    Returns:
        TrueEmotionPro: 实例
    """
    return TrueEmotionPro(memory_path=memory_path)
