"""
人性化情感检测器 v1.15
======================
支持:
- 情感复合检测
- 连续强度计算
- 上下文感知
- 微表情识别(标点、语气词)
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from trueemotion.core.emotions.plutchik24 import (
    EMOTION_VAD,
    EMOTION_KEYWORDS,
    EMOTION_ANTONYM,
    EMOTION_TRANSITIONS,
    calculate_compound_emotion,
    get_intensity_label,
)


@dataclass
class EmotionMatch:
    """情感匹配结果"""
    emotion: str
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    is_negated: bool = False


@dataclass
class DetectionContext:
    """检测上下文"""
    has_question: bool = False      # 问号
    has_exclamation: bool = False   # 感叹号
    has_ellipsis: bool = False      # 省略号
    has_repetition: bool = False    # 重复（！！或。。）
    sentence_length: int = 0        # 句子长度
    is_first_person: bool = False   # 第一人称


class HumanEmotionDetector:
    """
    人性化情感检测器

    特点:
    1. 连续强度 - 0.0到1.0的连续分数，不是离散标签
    2. 情感复合 - 检测多种情感的组合
    3. 上下文感知 - 考虑标点、语气词、句式
    4. 否定处理 - "不开心"不完全等于"开心"
    5. 强度修饰 - "太开心了"比"有点开心"强
    """

    # 否定词 (单独使用或作为完整否定词时)
    NEGATIONS = {
        "不", "不是", "没", "没有", "不会", "不要", "别", "莫", "非", "未", "休", "甭",
        "不太", "不怎么", "不太想", "不怎么想", "不太想", "不用", "并非",
        # 注意："无"作为单独字时是否定，但"无敌"中的"无"不是
    }

    # 不应该被当作否定的词组
    NEGATION_EXCEPTIONS = {
        "无敌", "无比", "无双", "无与伦比", "无疆", "无限", "无心",
        "无比幸福", "无比开心", "无比快乐",
    }

    # 强化词及倍数
    INTENSIFIERS: Dict[str, float] = {
        "太": 1.6, "好": 1.4, "真": 1.5, "非常": 1.5, "特别": 1.5,
        "极其": 1.8, "格外": 1.6, "十分": 1.5, "超": 1.6, "巨": 1.6,
        "贼": 1.5, "超级": 1.6, "相当": 1.4, "灰常": 1.5, "暴": 1.7,
        "无比": 1.8, "超级无敌": 2.0, "无敌": 1.8,
        "一点": 0.4, "有点": 0.6, "稍微": 0.5, "略微": 0.5, "稍有": 0.5,
        "比较": 0.8, "蛮": 0.7, "还挺": 0.7, "还": 0.5,
    }

    # 语气词（表达细腻情感）
    PARTICLES: Dict[str, float] = {
        "啊": 1.2, "呀": 1.15, "哇": 1.3, "哦": 1.1, "嗯": 0.9,
        "呢": 1.0, "吧": 0.8, "啦": 1.1, "嘛": 0.9, "哈": 1.15,
        "呃": 0.8, "唉": 0.9, "哎": 1.1, "哟": 1.2,
    }

    # 颜文字/表情
    EMOTICONS: Dict[str, str] = {
        ":)": "joy", ":-)": "joy", ":D": "joy", "=)": "joy",
        ":(": "sadness", ":-(": "sadness", ":d": "joy",
        ":o": "surprise", ":O": "surprise",
        ">:(": "anger", ">:)": "contempt",
        "<3": "love", ":*": "love", ";-)": "amusement", "XD": "amusement",
    }

    # 标点权重
    PUNCTUATION_WEIGHTS = {
        "!!!": 1.5, "！！": 1.5,
        "...": 0.8, "。。。": 0.8,  # 省略号暗示深沉情感
        "?!": 1.3, "！？": 1.3,     # 惊讶+疑问
    }

    def __init__(self, threshold: float = 0.03, evolved_rules: list = None):
        """
        初始化检测器

        Args:
            threshold: 检测阈值，低于此值的情感被过滤
            evolved_rules: 进化规则列表
        """
        self.threshold = threshold
        self._evolved_rules = evolved_rules or []
        self._build_index()
        self._apply_evolved_rules()

    def _build_index(self) -> None:
        """构建关键词索引以加速匹配"""
        self._keyword_to_emotions: Dict[str, List[Tuple[str, str]]] = {}

        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword not in self._keyword_to_emotions:
                    self._keyword_to_emotions[keyword] = []
                self._keyword_to_emotions[keyword].append((emotion, keyword))

    def _apply_evolved_rules(self) -> None:
        """应用进化规则到关键词索引"""
        for rule in self._evolved_rules:
            emotion = rule.get("emotion")
            keywords = rule.get("keywords", [])
            if emotion and keywords:
                for keyword in keywords:
                    if keyword not in self._keyword_to_emotions:
                        self._keyword_to_emotions[keyword] = []
                    self._keyword_to_emotions[keyword].append((emotion, keyword))

    def detect(self, text: str) -> Dict[str, float]:
        """
        检测情感

        Args:
            text: 输入文本

        Returns:
            Dict[str, float]: 情感及其连续强度分数 (0.0-1.0)
        """
        if not text or not text.strip():
            return {}

        text = text.strip()
        context = self._analyze_context(text)
        matches = self._find_matches(text)
        scores = self._calculate_continuous_scores(matches, context, text)

        # 计算复合情感
        compounds = calculate_compound_emotion(scores)

        # 合并
        all_emotions = {**scores, **compounds}

        # 归一化并过滤
        return self._normalize_and_filter(all_emotions)

    def _analyze_context(self, text: str) -> DetectionContext:
        """分析文本上下文"""
        context = DetectionContext()

        # 标点分析
        context.has_question = "？" in text or "?" in text
        context.has_exclamation = "！" in text or "!" in text
        context.has_ellipsis = "..." in text or "。。。" in text or "……" in text

        # 重复检测
        context.has_repetition = (
            text.count("!") >= 2 or text.count("！") >= 2 or
            text.count(".") >= 3 or text.count("。") >= 3
        )

        # 句长
        context.sentence_length = len(text)

        # 人称
        context.is_first_person = any(p in text for p in ["我", "我们", "俺", "咱"])

        return context

    def _find_matches(self, text: str) -> List[EmotionMatch]:
        """查找所有情感匹配"""
        matches: List[EmotionMatch] = []

        # 1. 关键词匹配
        for keyword, emotion_list in self._keyword_to_emotions.items():
            if keyword in text:
                for emotion, exact_keyword in emotion_list:
                    is_negated = self._check_negation(text, keyword)
                    base_score = 0.12

                    match = EmotionMatch(
                        emotion=emotion,
                        score=base_score,
                        matched_keywords=[exact_keyword],
                        is_negated=is_negated,
                    )
                    matches.append(match)

        # 2. 颜文字匹配
        for emoticon, emotion in self.EMOTICONS.items():
            if emoticon in text:
                matches.append(EmotionMatch(
                    emotion=emotion,
                    score=0.15,
                    matched_keywords=[emoticon],
                ))

        # 3. 情感词组匹配（成语、习惯用语）
        phrases = self._find_phrases(text)
        matches.extend(phrases)

        return matches

    def _find_phrases(self, text: str) -> List[EmotionMatch]:
        """查找情感词组"""
        matches: List[EmotionMatch] = []

        # 常见情感词组
        phrase_emotions = {
            "心如刀割": "grief",
            "撕心裂肺": "grief",
            "悲痛欲绝": "grief",
            "欢天喜地": "ecstasy",
            "心花怒放": "ecstasy",
            "暴跳如雷": "rage",
            "火冒三丈": "rage",
            "忐忑不安": "anxiety",
            "如坐针毡": "anxiety",
            "喜极而泣": "bittersweet",
            "百感交集": "confusion",
            "感同身受": "compassion",
            "感激不尽": "gratitude",
            "五味杂陈": "confusion",
            "哭笑不得": "bittersweet",
            "乐极生悲": "bittersweet",
            "怒其不争": "regret",
            "哀其不幸": "compassion",
        }

        for phrase, emotion in phrase_emotions.items():
            if phrase in text:
                matches.append(EmotionMatch(
                    emotion=emotion,
                    score=0.20,
                    matched_keywords=[phrase],
                ))

        return matches

    def _check_negation(self, text: str, keyword: str) -> bool:
        """检查关键词是否被否定"""
        try:
            idx = text.index(keyword)
            # 检查前面的60个字符
            start = max(0, idx - 60)
            before_keyword = text[start:idx]

            # 检查是否有例外词组（如"无敌"）
            for exception in self.NEGATION_EXCEPTIONS:
                if exception in before_keyword or exception in text:
                    return False

            for neg in self.NEGATIONS:
                if neg in before_keyword:
                    return True
        except ValueError:
            pass
        return False

    def _calculate_continuous_scores(
        self,
        matches: List[EmotionMatch],
        context: DetectionContext,
        text: str,
    ) -> Dict[str, float]:
        """计算连续强度分数"""
        scores: Dict[str, float] = {}

        # 聚合相同情感的匹配
        emotion_matches: Dict[str, List[EmotionMatch]] = {}
        for match in matches:
            if match.emotion not in emotion_matches:
                emotion_matches[match.emotion] = []
            emotion_matches[match.emotion].append(match)

        for emotion, emotion_match_list in emotion_matches.items():
            score = 0.0
            all_keywords: List[str] = []

            for match in emotion_match_list:
                if match.is_negated:
                    # 否定情感降低影响，但不完全消除（人有"口是心非"）
                    score += match.score * 0.25
                else:
                    score += match.score
                all_keywords.extend(match.matched_keywords)

            # 应用上下文权重
            score = self._apply_context_weights(score, emotion, context, text, all_keywords)

            scores[emotion] = min(1.0, score)

        return scores

    def _apply_context_weights(
        self,
        score: float,
        emotion: str,
        context: DetectionContext,
        text: str,
        keywords: List[str],
    ) -> float:
        """应用上下文权重"""

        # 1. 感叹号强化
        if context.has_exclamation:
            exclamations = min(text.count("！") + text.count("!"), 3)
            if exclamations > 0:
                boost = 1.0 + (0.15 * exclamations)
                score *= boost

        # 2. 问号降低（疑问句情感表达相对弱）
        if context.has_question:
            score *= 0.85

        # 3. 省略号暗示更深沉情感
        if context.has_ellipsis and score > 0.1:
            score *= 1.2

        # 4. 重复标点强化
        if context.has_repetition:
            score *= 1.3

        # 5. 句长归一化（太短可能误判，太长稀释）
        if context.sentence_length < 5:
            score *= 0.7
        elif context.sentence_length > 50:
            score *= 0.9

        # 6. 强化词
        matched_multipliers = []
        for word, multiplier in self.INTENSIFIERS.items():
            if word in text:
                matched_multipliers.append(multiplier)
        if matched_multipliers:
            best = max(matched_multipliers, key=lambda m: abs(m - 1.0))
            score *= best

        # 7. 语气词
        for particle, multiplier in self.PARTICLES.items():
            if particle in text:
                score *= multiplier

        # 8. 情感特定的上下文调整
        if emotion in ["joy", "ecstasy"] and context.has_exclamation:
            score *= 1.2  # 正面情感配感叹号加强

        if emotion in ["sadness", "grief"] and context.has_ellipsis:
            score *= 1.3  # 悲伤配省略号加强

        if emotion in ["anger", "rage"] and context.has_repetition:
            score *= 1.4  # 愤怒配重复加强

        return min(1.0, score)

    def _normalize_and_filter(self, scores: Dict[str, float]) -> Dict[str, float]:
        """归一化并过滤"""
        if not scores:
            return {}

        # 过滤低分
        filtered = {k: v for k, v in scores.items() if v >= self.threshold}

        if not filtered:
            return {}

        # 归一化到0-1
        max_score = max(filtered.values())
        if max_score > 1.0:
            filtered = {k: v / max_score for k, v in filtered.items()}

        # 按分数降序排序
        return dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))

    def get_top_emotions(self, text: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        获取Top-K情感

        Returns:
            List[Tuple[情感, 分数, 强度标签]]
        """
        scores = self.detect(text)
        result = []
        for emotion, score in list(scores.items())[:top_k]:
            label = get_intensity_label(score)
            result.append((emotion, round(score, 3), label))
        return result

    def explain(self, text: str) -> Dict:
        """解释检测结果"""
        scores = self.detect(text)
        context = self._analyze_context(text)

        matches = self._find_matches(text)
        emotion_matches = {}
        for m in matches:
            if m.emotion not in emotion_matches:
                emotion_matches[m.emotion] = []
            emotion_matches[m.emotion].append({
                "keyword": m.matched_keywords,
                "negated": m.is_negated,
            })

        return {
            "text": text,
            "detected_emotions": {k: round(v, 3) for k, v in scores.items()},
            "context": {
                "has_question": context.has_question,
                "has_exclamation": context.has_exclamation,
                "has_ellipsis": context.has_ellipsis,
                "sentence_length": context.sentence_length,
                "is_first_person": context.is_first_person,
            },
            "matches": emotion_matches,
        }
