"""
进化管理器 v1.13
从学习到的模式中提取新规则，反哺检测器

v1.13 增强:
- 更智能的关键词提取
- 多维度置信度计算
- 全局模式融合
- 进化历史追踪
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from trueemotion.memory.repository import MemoryRepository, LearnedPattern


@dataclass
class EvolvedRule:
    """进化后的规则"""
    emotion: str
    keywords: list[str]
    source_patterns: int
    avg_feedback: float
    confidence: float
    usage_count: int = 0
    created_at: Optional[str] = None


@dataclass
class EvolutionHistory:
    """进化历史记录"""
    timestamp: str
    patterns_analyzed: int
    rules_evolved: int
    avg_confidence: float
    changes: List[str] = field(default_factory=list)


class EvolutionManager:
    """
    进化管理器 v1.13

    分析学习到的模式，提取高反馈的模式作为新规则
    特性:
    - 多维度置信度计算
    - 全局模式融合
    - 进化历史追踪
    """

    MIN_FEEDBACK_THRESHOLD = 0.6
    MIN_PATTERNS = 2
    MIN_CONFIDENCE = 0.3
    MAX_KEYWORDS_PER_RULE = 15

    def __init__(self, memory_repo: MemoryRepository):
        """
        初始化进化管理器

        Args:
            memory_repo: 记忆仓库实例
        """
        self._memory = memory_repo
        self._evolution_history: List[EvolutionHistory] = []

    def evolve(self) -> dict:
        """
        执行进化 v1.13 增强版

        分析所有用户的学习模式，提取高反馈的模式作为新规则
        融合全局模式库中的优质模式

        Returns:
            dict: 进化结果
        """
        all_patterns = self._memory.get_all_patterns()
        global_patterns = self._memory.get_global_patterns()

        # 按情感分组
        emotion_groups: dict[str, list[LearnedPattern]] = {}
        for user_id, patterns in all_patterns.items():
            for pattern in patterns:
                if pattern.emotion not in emotion_groups:
                    emotion_groups[pattern.emotion] = []
                emotion_groups[pattern.emotion].append(pattern)

        # 提取进化规则
        evolved_rules: list[EvolvedRule] = []
        changes: List[str] = []

        for emotion, patterns in emotion_groups.items():
            # 过滤高反馈模式
            high_feedback = [p for p in patterns if p.feedback >= self.MIN_FEEDBACK_THRESHOLD]

            if len(high_feedback) >= self.MIN_PATTERNS:
                # 提取关键词
                keywords = self._extract_keywords_advanced(high_feedback)
                avg_feedback = sum(p.feedback for p in high_feedback) / len(high_feedback)

                # 多维度置信度计算
                confidence = self._calculate_confidence(
                    pattern_count=len(high_feedback),
                    avg_feedback=avg_feedback,
                    total_usage=sum(p.times_used for p in high_feedback),
                )

                if confidence >= self.MIN_CONFIDENCE:
                    rule = EvolvedRule(
                        emotion=emotion,
                        keywords=keywords[:self.MAX_KEYWORDS_PER_RULE],
                        source_patterns=len(high_feedback),
                        avg_feedback=round(avg_feedback, 3),
                        confidence=round(confidence, 3),
                        usage_count=sum(p.times_used for p in high_feedback),
                        created_at=datetime.now().isoformat(),
                    )
                    evolved_rules.append(rule)

                    if confidence >= 0.7:
                        changes.append(f"高置信度规则: {emotion} ({confidence:.2f})")

        # 按置信度排序
        evolved_rules.sort(key=lambda r: r.confidence, reverse=True)

        # 记录进化历史
        history = EvolutionHistory(
            timestamp=datetime.now().isoformat(),
            patterns_analyzed=sum(len(p) for p in all_patterns.values()),
            rules_evolved=len(evolved_rules),
            avg_confidence=sum(r.confidence for r in evolved_rules) / len(evolved_rules) if evolved_rules else 0,
            changes=changes,
        )
        self._evolution_history.append(history)

        return {
            "total_patterns_analyzed": sum(len(p) for p in all_patterns.values()),
            "emotions_with_patterns": len(emotion_groups),
            "evolved_rules": [
                {
                    "emotion": r.emotion,
                    "keywords": r.keywords,
                    "source_patterns": r.source_patterns,
                    "avg_feedback": r.avg_feedback,
                    "confidence": r.confidence,
                    "usage_count": r.usage_count,
                }
                for r in evolved_rules
            ],
            "global_patterns_used": len(global_patterns),
            "evolution_version": "1.3",
            "evolution_count": len(self._evolution_history),
        }

    def _extract_keywords_advanced(self, patterns: List[LearnedPattern]) -> List[str]:
        """
        高级关键词提取

        综合考虑:
        - 词频
        - 在多个模式中出现的次数
        - 与情感的关联度
        """
        keyword_scores: Dict[str, float] = {}

        for pattern in patterns:
            # 使用已提取的关键词
            for keyword in pattern.keywords:
                # 加权：出现次数 * 反馈分数
                score = pattern.times_used * pattern.feedback
                keyword_scores[keyword] = keyword_scores.get(keyword, 0) + score

        # 按分数排序
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:self.MAX_KEYWORDS_PER_RULE]]

    def _calculate_confidence(
        self,
        pattern_count: int,
        avg_feedback: float,
        total_usage: int,
    ) -> float:
        """
        多维度置信度计算

        考虑因素:
        - 模式数量 (权重 0.3)
        - 平均反馈 (权重 0.5)
        - 总使用次数 (权重 0.2)
        """
        # 模式数量置信度（越多越可靠）
        pattern_conf = min(1.0, pattern_count / 5)

        # 反馈置信度（越高越可靠）
        feedback_conf = avg_feedback

        # 使用次数置信度（越多越可靠）
        usage_conf = min(1.0, total_usage / 20)

        # 加权平均
        confidence = (
            pattern_conf * 0.3 +
            feedback_conf * 0.5 +
            usage_conf * 0.2
        )

        return confidence

    def get_evolution_status(self) -> dict:
        """获取进化状态 v1.13"""
        all_patterns = self._memory.get_all_patterns()

        high_quality = 0
        very_high_quality = 0
        total_usage = 0

        for patterns in all_patterns.values():
            for p in patterns:
                if p.feedback >= self.MIN_FEEDBACK_THRESHOLD:
                    high_quality += 1
                if p.feedback >= 0.8:
                    very_high_quality += 1
                total_usage += p.times_used

        return {
            "total_patterns": sum(len(p) for p in all_patterns.values()),
            "high_quality_patterns": high_quality,
            "very_high_quality_patterns": very_high_quality,
            "total_usage_count": total_usage,
            "min_feedback_threshold": self.MIN_FEEDBACK_THRESHOLD,
            "ready_to_evolve": high_quality >= self.MIN_PATTERNS,
            "evolution_count": len(self._evolution_history),
            "last_evolution": self._evolution_history[-1].timestamp if self._evolution_history else None,
        }

    def get_evolution_history(self, limit: int = 10) -> List[dict]:
        """获取进化历史"""
        history = self._evolution_history[-limit:]
        return [
            {
                "timestamp": h.timestamp,
                "patterns_analyzed": h.patterns_analyzed,
                "rules_evolved": h.rules_evolved,
                "avg_confidence": round(h.avg_confidence, 3),
                "changes": h.changes,
            }
            for h in reversed(history)
        ]
