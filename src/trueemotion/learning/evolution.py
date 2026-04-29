"""
进化管理器
从学习到的模式中提取新规则，反哺检测器
"""

from dataclasses import dataclass
from typing import Optional
from trueemotion.memory.repository import MemoryRepository, LearnedPattern


@dataclass
class EvolvedRule:
    """进化后的规则"""
    emotion: str
    keywords: list[str]
    source_patterns: int
    avg_feedback: float
    confidence: float


class EvolutionManager:
    """
    进化管理器

    分析学习到的模式，提取高反馈的模式作为新规则
    """

    MIN_FEEDBACK_THRESHOLD = 0.6
    MIN_PATTERNS = 2

    def __init__(self, memory_repo: MemoryRepository):
        """
        初始化进化管理器

        Args:
            memory_repo: 记忆仓库实例
        """
        self._memory = memory_repo

    def evolve(self) -> dict:
        """
        执行进化

        分析所有用户的学习模式，提取高反馈的模式作为新规则建议

        Returns:
            dict: 进化结果
        """
        all_patterns = self._memory.get_all_patterns()

        # 按情感分组
        emotion_groups: dict[str, list[LearnedPattern]] = {}
        for user_id, patterns in all_patterns.items():
            for pattern in patterns:
                if pattern.emotion not in emotion_groups:
                    emotion_groups[pattern.emotion] = []
                emotion_groups[pattern.emotion].append(pattern)

        # 提取进化规则
        evolved_rules: list[EvolvedRule] = []
        for emotion, patterns in emotion_groups.items():
            # 过滤高反馈模式
            high_feedback = [p for p in patterns if p.feedback >= self.MIN_FEEDBACK_THRESHOLD]

            if len(high_feedback) >= self.MIN_PATTERNS:
                # 提取关键词（从响应中提取特征词）
                keywords = self._extract_keywords(high_feedback)
                avg_feedback = sum(p.feedback for p in high_feedback) / len(high_feedback)
                confidence = min(1.0, len(high_feedback) / 5)  # 最多5个pattern达到1.0

                evolved_rules.append(EvolvedRule(
                    emotion=emotion,
                    keywords=keywords[:10],  # 最多保留10个关键词
                    source_patterns=len(high_feedback),
                    avg_feedback=avg_feedback,
                    confidence=confidence,
                ))

        # 按置信度排序
        evolved_rules.sort(key=lambda r: r.confidence, reverse=True)

        return {
            "total_patterns_analyzed": sum(len(p) for p in all_patterns.values()),
            "emotions_with_patterns": len(emotion_groups),
            "evolved_rules": [
                {
                    "emotion": r.emotion,
                    "keywords": r.keywords,
                    "source_patterns": r.source_patterns,
                    "avg_feedback": round(r.avg_feedback, 2),
                    "confidence": round(r.confidence, 2),
                }
                for r in evolved_rules
            ],
            "evolution_version": "1.0",
        }

    def _extract_keywords(self, patterns: list[LearnedPattern]) -> list[str]:
        """从模式中提取关键词"""
        # 简单实现：收集所有响应文本中的词
        # 实际应该使用更复杂的NLP技术
        word_freq: dict[str, int] = {}

        for pattern in patterns:
            words = pattern.response.replace(",", " ").replace("。", " ").replace("！", " ").split()
            for word in words:
                if len(word) >= 2:  # 忽略单字
                    word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    def get_evolution_status(self) -> dict:
        """获取进化状态"""
        all_patterns = self._memory.get_all_patterns()

        high_quality = 0
        for patterns in all_patterns.values():
            high_quality += sum(1 for p in patterns if p.feedback >= self.MIN_FEEDBACK_THRESHOLD)

        return {
            "total_patterns": sum(len(p) for p in all_patterns.values()),
            "high_quality_patterns": high_quality,
            "min_feedback_threshold": self.MIN_FEEDBACK_THRESHOLD,
            "ready_to_evolve": high_quality >= self.MIN_PATTERNS,
        }
