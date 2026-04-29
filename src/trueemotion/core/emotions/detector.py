"""
基于规则的情感检测器
使用关键词匹配和VAD模型进行情感分析
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from trueemotion.core.emotions.plutchik24 import EMOTION_VAD, EMOTION_KEYWORDS


@dataclass
class EmotionMatch:
    """情感匹配结果"""
    emotion: str
    score: float
    matched_keywords: list[str] = field(default_factory=list)


class RuleBasedEmotionDetector:
    """
    基于规则的情感检测器

    使用方法:
        detector = RuleBasedEmotionDetector()
        result = detector.detect("今天太开心了！")
    """

    # 否定词列表
    NEGATIONS = {"不", "不是", "没", "没有", "不会", "不要", "别", "莫", "非", "无", "未", "休", "甭"}

    # 强化词列表
    INTENSIFIERS = {
        "太": 1.5, "好": 1.3, "真": 1.3, "非常": 1.4, "特别": 1.4,
        "极其": 1.6, "格外": 1.5, "十分": 1.4, "超": 1.5, "巨": 1.5,
        "贼": 1.4, "超级": 1.5, "非常": 1.4, "相当": 1.3,
        "一点": 0.7, "有点": 0.8, "稍微": 0.7, "略微": 0.7, "稍有": 0.7,
    }

    # 感叹号数量强化
    EXCLAMATION_BOOST = 1.2
    QUESTION_REDUCE = 0.9

    def __init__(self, threshold: float = 0.05):
        """
        初始化检测器

        Args:
            threshold: 情感检测阈值，低于此值的情感会被过滤
        """
        self.threshold = threshold
        self._build_patterns()

    def _build_patterns(self) -> None:
        """预编译正则表达式模式"""
        # 组合所有关键词为正则模式
        all_keywords: set[str] = set()
        for keywords in EMOTION_KEYWORDS.values():
            all_keywords.update(keywords)

        # 创建高效匹配的关键词模式
        self._keyword_pattern = re.compile(
            "|".join(re.escape(kw) for kw in all_keywords)
        )

    def detect(self, text: str) -> dict[str, float]:
        """
        检测文本中的情感

        Args:
            text: 输入文本

        Returns:
            dict[str, float]: 情感及其得分字典
        """
        if not text or not text.strip():
            return {}

        text = text.strip()
        matches = self._find_emotion_matches(text)
        scores = self._calculate_scores(matches, text)
        return self._filter_and_normalize(scores)

    def _find_emotion_matches(self, text: str) -> list[EmotionMatch]:
        """查找文本中的情感匹配"""
        matches: list[EmotionMatch] = []

        for emotion, keywords in EMOTION_KEYWORDS.items():
            emotion_matches: list[str] = []
            total_score = 0.0

            for keyword in keywords:
                if keyword in text:
                    # 检查是否被否定
                    is_negated = self._is_negated(text, keyword)
                    base_score = 0.15  # 每个关键词基础分数

                    if is_negated:
                        base_score *= -0.5  # 否定词降低影响
                    else:
                        emotion_matches.append(keyword)
                        total_score += base_score

            if emotion_matches:
                # 应用文本长度归一化
                length_factor = min(1.0, len(text) / 20)
                matches.append(EmotionMatch(
                    emotion=emotion,
                    score=total_score * length_factor,
                    matched_keywords=emotion_matches
                ))

        return matches

    def _is_negated(self, text: str, keyword: str) -> bool:
        """检查关键词是否被否定"""
        try:
            idx = text.index(keyword)
            # 检查前面的50个字符
            start = max(0, idx - 50)
            before_keyword = text[start:idx]

            for neg in self.NEGATIONS:
                if neg in before_keyword:
                    return True
        except ValueError:
            pass
        return False

    def _calculate_scores(self, matches: list[EmotionMatch], text: str) -> dict[str, float]:
        """计算情感得分"""
        scores: dict[str, float] = {}

        for match in matches:
            # 基础分数
            score = match.score

            # 应用感叹号强化
            exclamations = text.count("！") + text.count("!")
            if exclamations > 0:
                score *= (self.EXCLAMATION_BOOST ** min(exclamations, 3))

            # 应用问号降低
            questions = text.count("？") + text.count("?")
            if questions > 0:
                score *= (self.QUESTION_REDUCE ** min(questions, 3))

            # 应用强化词
            for intensifier, factor in self.INTENSIFIERS.items():
                if intensifier in text:
                    score *= factor
                    break

            # 检查否定反转
            if any(neg in text for neg in self.NEGATIONS):
                if match.emotion in ("joy", "trust", "love", "admiration"):
                    score *= -0.3  # 正面情感被否定

            scores[match.emotion] = scores.get(match.emotion, 0) + score

        return scores

    def _filter_and_normalize(self, scores: dict[str, float]) -> dict[str, float]:
        """过滤低分情感并归一化"""
        if not scores:
            return {}

        # 过滤低于阈值的
        filtered = {k: v for k, v in scores.items() if v >= self.threshold}

        if not filtered:
            return {}

        # 归一化到0-1
        max_score = max(filtered.values())
        if max_score > 1.0:
            filtered = {k: v / max_score for k, v in filtered.items()}

        # 按分数排序
        return dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))

    def get_top_emotions(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        获取top-k情感

        Args:
            text: 输入文本
            top_k: 返回数量

        Returns:
            list[tuple[str, float]]: 情感和分数元组列表
        """
        scores = self.detect(text)
        return list(scores.items())[:top_k]

    def get_vad(self, emotion: str) -> Optional[tuple[float, float, float]]:
        """获取情感的VAD坐标"""
        return EMOTION_VAD.get(emotion)
