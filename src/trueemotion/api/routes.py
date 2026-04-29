"""
TrueEmotion Pro v1.14 API
人性化情感AI系统
"""

import os
from typing import Optional, List

from trueemotion.core.analysis.analyzer import EmotionAnalyzer, AnalyzeOptions
from trueemotion.core.analysis.output import AnalysisResult
from trueemotion.learning.evolution import EvolutionManager
from trueemotion.memory.repository import MemoryRepository

# LLM 组件（可选）
try:
    from trueemotion.core.llm import OpenAIClient, LLM_AVAILABLE
except ImportError:
    LLM_AVAILABLE = False
    OpenAIClient = None


class TrueEmotionPro:
    """
    TrueEmotion Pro v1.14 主类

    使用方法:
        pro = TrueEmotionPro()
        result = pro.analyze("今天太开心了！")
        print(result.emotion.primary)  # joy
        print(result.human_response.text)  # "太为你高兴了！说说怎么回事！"

    v1.14 新增:
        pro = TrueEmotionPro(llm_provider="openai", api_key="sk-...")  # 启用 LLM
        result = pro.analyze("今天被老板画饼了...")  # LLM 理解深层语义
    """

    def __init__(
        self,
        memory_path: str = "./memory",
        auto_learn: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        enable_llm: bool = True,
    ):
        """
        初始化TrueEmotion Pro

        Args:
            memory_path: 记忆存储路径
            auto_learn: 是否自动学习用户反馈
            llm_provider: LLM Provider ("openai" 或 None)
            llm_model: LLM 模型名称
            api_key: API Key (默认从环境变量获取)
            enable_llm: 是否启用 LLM (当 llm_provider 提供时生效)
        """
        self._memory = MemoryRepository(memory_path)

        # 初始化 LLM 客户端
        llm_client = None
        if LLM_AVAILABLE and llm_provider:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                llm_client = OpenAIClient(
                    api_key=api_key,
                    model=llm_model,
                )
                self._llm_client = llm_client
            else:
                import logging
                logging.warning("LLM provider specified but no API key found")
        else:
            self._llm_client = None

        self._analyzer = EmotionAnalyzer(
            memory_path=memory_path,
            detector=None,
            empathy_engine=None,
            llm_client=llm_client,
            enable_llm=enable_llm and llm_client is not None,
        )
        self._evolution = EvolutionManager(self._memory)
        self._auto_learn = auto_learn

    @property
    def is_llm_enabled(self) -> bool:
        """是否启用了 LLM"""
        return self._analyzer.is_llm_enabled

    @property
    def is_llm_available(self) -> bool:
        """LLM 是否可用"""
        return self._analyzer.is_llm_available

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

    def analyze_batch(
        self,
        texts: List[str],
        user_id: str = "default",
    ) -> List[AnalysisResult]:
        """
        批量分析文本情感

        Args:
            texts: 输入文本列表
            user_id: 用户ID

        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        return [self.analyze(text, user_id=user_id) for text in texts]

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
        stats["version"] = "1.14"
        stats["engine"] = "llm-v1.14" if self.is_llm_enabled else "rule-v1.14"
        stats["evolution"] = self._evolution.get_evolution_status()
        return stats


# 便捷函数
def create_analyzer(
    memory_path: str = "./memory",
    llm_provider: Optional[str] = None,
    **kwargs,
) -> TrueEmotionPro:
    """
    创建TrueEmotion Pro实例

    Args:
        memory_path: 记忆存储路径
        llm_provider: LLM Provider ("openai" 或 None)
        **kwargs: 其他参数传递给 TrueEmotionPro

    Returns:
        TrueEmotionPro: 实例
    """
    return TrueEmotionPro(
        memory_path=memory_path,
        llm_provider=llm_provider,
        **kwargs,
    )
