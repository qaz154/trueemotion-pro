"""
情感分析器门面 v1.13
====================
整合检测器、记忆系统、响应生成器

v1.13 新增:
- 反讽检测
- 上下文理解
- 主动共情
"""

from dataclasses import dataclass
from typing import Optional, Dict

from trueemotion.core.emotions.detector import HumanEmotionDetector
from trueemotion.core.emotions.plutchik24 import (
    EMOTION_VAD,
    get_intensity_label,
    get_vad_label,
)
from trueemotion.core.emotions.irony import IronyDetector, IronyResult
from trueemotion.core.analysis.context import ContextualAnalyzer, ConversationContext
from trueemotion.core.analysis.output import (
    EmotionOutput,
    HumanResponse,
    UserProfile,
    AnalysisResult,
)
from trueemotion.core.response.engine import HumanEmpathyEngine
from trueemotion.memory.repository import MemoryRepository


@dataclass
class AnalyzeOptions:
    """分析选项"""
    learn: bool = False
    response: Optional[str] = None
    feedback: float = 0.5
    user_id: str = "default"
    context: Optional[str] = None


class EmotionAnalyzer:
    """
    情感分析器门面

    v1.13 新特性:
    - 人性化情感检测（连续强度、复合情感）
    - 更细腻的共情响应
    - 上下文感知
    - 反讽检测
    - 主动共情
    """

    def __init__(
        self,
        memory_path: str = "./memory",
        detector: Optional[HumanEmotionDetector] = None,
        empathy_engine: Optional[HumanEmpathyEngine] = None,
    ):
        self._detector = detector or HumanEmotionDetector()
        self._empathy = empathy_engine or HumanEmpathyEngine()
        self._memory = MemoryRepository(memory_path)
        self._irony = IronyDetector()
        self._context_analyzer = ContextualAnalyzer()
        self._conversation_contexts: Dict[str, ConversationContext] = {}

    def analyze(self, text: str, options: Optional[AnalyzeOptions] = None) -> AnalysisResult:
        """
        分析文本情感

        Args:
            text: 输入文本
            options: 分析选项

        Returns:
            AnalysisResult: 完整分析结果
        """
        opts = options or AnalyzeOptions(text=text)

        # 1. 情感检测（人性化）
        emotion_scores = self._detector.detect(text)

        # 2. 获取主要情感和详细信息
        primary_emotion, primary_score = self._get_primary(emotion_scores)
        vad = EMOTION_VAD.get(primary_emotion, (0.0, 0.0, 0.0))
        intensity_label = get_intensity_label(primary_score)

        # 3. 反讽检测
        irony_result = self._irony.detect(text, primary_emotion, primary_score)
        effective_emotion = irony_result.true_emotion or primary_emotion
        effective_intensity = primary_score

        # 如果检测到反讽，使用真实情感
        if irony_result.is_irony:
            effective_emotion = irony_result.true_emotion

        # 4. 上下文分析
        context_result = self._context_analyzer.analyze_with_context(
            text, effective_emotion, effective_intensity
        )

        # 5. 分离纯情感和复合情感
        pure_emotions = {k: v for k, v in emotion_scores.items()
                        if not self._is_compound_emotion(k)}
        compound_emotions = {k: v for k, v in emotion_scores.items()
                           if self._is_compound_emotion(k)}

        # 6. 生成共情回复
        human_response = self._empathy.generate(
            emotion=effective_emotion,
            intensity=effective_intensity,
            context=opts.context,
        )

        # 7. 如果需要追问，使用上下文分析结果
        follow_up_suggestion = self._context_analyzer.get_follow_up_suggestion(
            effective_emotion, effective_intensity, context_result
        )
        if follow_up_suggestion and not human_response.follow_up:
            human_response.follow_up = follow_up_suggestion

        # 8. 构建情感混合描述
        emotion_mix = self._build_emotion_mix(emotion_scores)

        # 9. 更新用户记忆
        user_profile = self._update_memory(
            user_id=opts.user_id,
            emotion=effective_emotion,
            learn=opts.learn,
            response=opts.response,
            feedback=opts.feedback,
        )

        # 10. 获取检测解释
        explanation = self._detector.explain(text) if primary_score > 0.1 else None

        # 构建explanation加入反讽信息
        if irony_result.is_irony:
            if explanation is None:
                explanation = {}
            explanation["irony"] = {
                "is_irony": True,
                "surface_emotion": irony_result.surface_emotion,
                "true_emotion": irony_result.true_emotion,
                "confidence": irony_result.confidence,
                "clues": irony_result.clues,
            }

        return AnalysisResult(
            version="1.11",
            engine="humanized-v1.13",
            emotion=EmotionOutput(
                primary=effective_emotion,
                intensity=effective_intensity,
                vad=vad,
                confidence=primary_score,
                intensity_label=intensity_label,
                all_emotions=emotion_scores,
                compound_emotions=compound_emotions,
                emotion_mix=self._build_top_emotions_list(emotion_scores),
            ),
            human_response=HumanResponse(
                text=human_response.text,
                empathy_type=human_response.empathy_type,
                intensity_level=human_response.intensity_level,
                follow_up=human_response.follow_up,
                empathy_depth=human_response.tone,
                tone=human_response.tone,
            ),
            user_profile=user_profile,
            context_used=opts.context is not None,
            emotion_mix=emotion_mix,
            explanation=explanation,
        )

    def _get_primary(self, scores: dict) -> tuple:
        """获取主要情感"""
        if not scores:
            return "neutral", 0.0
        primary = max(scores.items(), key=lambda x: x[1])
        return primary[0], primary[1]

    def _is_compound_emotion(self, emotion: str) -> bool:
        """判断是否为复合情感"""
        compound_emotions = {
            "bittersweet", "painful_joy", "happy_sadness",
            "love", "hope", "despair", "regret",
            "guilt", "pride", "envy", "contempt",
            "gratitude_love", "hope_fear", "jealous_love",
            "frustration_hopelessness", "love_admiration",
        }
        return emotion in compound_emotions

    def _build_emotion_mix(self, scores: dict) -> str:
        """构建情感混合描述"""
        if not scores:
            return "中性"

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_emotions) == 1:
            return f"以{self._get_emotion_cn(sorted_emotions[0][0])}为主"

        if len(sorted_emotions) == 2:
            e1, s1 = sorted_emotions[0]
            e2, s2 = sorted_emotions[1]
            return f"以{self._get_emotion_cn(e1)}为主，伴有{self._get_emotion_cn(e2)}"

        # 三个以上
        e1, s1 = sorted_emotions[0]
        e2, s2 = sorted_emotions[1]
        e3, s3 = sorted_emotions[2]
        return f"以{self._get_emotion_cn(e1)}为主，伴有{self._get_emotion_cn(e2)}和{self._get_emotion_cn(e3)}"

    def _build_top_emotions_list(self, scores: dict) -> list:
        """构建Top情感列表"""
        return [(e, round(s, 3)) for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]

    def _get_emotion_cn(self, emotion: str) -> str:
        """获取情感中文名"""
        emotion_cn = {
            "joy": "喜悦", "sadness": "悲伤", "anger": "愤怒", "fear": "恐惧",
            "disgust": "厌恶", "surprise": "惊讶", "trust": "信任",
            "anticipation": "期待", "ecstasy": "狂喜", "grief": "悲痛",
            "rage": "暴怒", "terror": "恐惧", "anxiety": "焦虑",
            "love": "爱", "hope": "希望", "despair": "绝望",
            "guilt": "内疚", "pride": "自豪", "envy": "嫉妒",
            "bittersweet": "悲喜交加", "neutral": "中性",
            "contentment": "满足", "melancholy": "忧郁", "loneliness": "孤独",
            "compassion": "同情", "gratitude": "感激", "regret": "遗憾",
            "pride": "骄傲", "confusion": "困惑", "nostalgia": "怀旧",
        }
        return emotion_cn.get(emotion, emotion)

    def _update_memory(
        self,
        user_id: str,
        emotion: str,
        learn: bool,
        response: Optional[str],
        feedback: float,
    ) -> UserProfile:
        """更新用户记忆"""
        profile = self._memory.get_user(user_id)

        profile.total_interactions += 1
        profile.last_emotion = emotion
        profile.emotional_history.append(emotion)

        # 更新主导情感
        if profile.dominant_emotion is None:
            profile.dominant_emotion = emotion

        # 学习新模式
        if learn and response:
            self._memory.learn_pattern(user_id, emotion, response, feedback)
            profile.learned_patterns = self._memory.get_pattern_count(user_id)

        # 保持历史长度
        if len(profile.emotional_history) > 100:
            profile.emotional_history = profile.emotional_history[-100:]

        self._memory.save_user(user_id, profile)

        return profile

    def get_user_profile(self, user_id: str) -> UserProfile:
        """获取用户画像"""
        return self._memory.get_user(user_id)

    def evolve(self) -> dict:
        """执行进化"""
        patterns = self._memory.get_all_patterns()
        return {
            "total_patterns": len(patterns),
            "patterns": patterns,
            "status": "evolved",
        }

    def get_stats(self) -> dict:
        """获取系统统计"""
        return self._memory.get_stats()
