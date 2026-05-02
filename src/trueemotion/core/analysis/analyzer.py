"""
情感分析器门面 v1.15
====================
整合检测器、记忆系统、响应生成器

v1.15 新增:
- LLM 驱动的语义情感检测
- LLM 驱动的动态响应生成
- 降级机制（LLM 不可用时自动切换到规则引擎）
- 反讽检测
- 上下文理解
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from trueemotion import __version__

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

# LLM 组件（可选）
try:
    from trueemotion.core.llm import (
        BaseLLMClient,
        OpenAIClient,
        LLMEmotionDetector,
        LLMResponseGenerator,
        FallbackManager,
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLMClient = None
    OpenAIClient = None
    LLMEmotionDetector = None
    LLMResponseGenerator = None
    FallbackManager = None


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
    情感分析器门面 v1.15

    v1.15 新特性:
    - LLM 驱动的语义情感检测（更准确的复杂情感理解）
    - LLM 驱动的动态响应生成（更个性化、口语化）
    - 规则引擎降级保障（LLM 不可用时自动切换）
    - 上下文感知
    - 反讽检测

    v1.15 特性:
    - 人性化情感检测（连续强度、复合情感）
    - 更细腻的共情响应
    - 主动共情
    """

    def __init__(
        self,
        memory_path: str = "./memory",
        detector: Optional[HumanEmotionDetector] = None,
        empathy_engine: Optional[HumanEmpathyEngine] = None,
        llm_client: Optional[BaseLLMClient] = None,
        enable_llm: bool = True,
    ):
        """
        初始化情感分析器

        Args:
            memory_path: 记忆存储路径
            detector: 规则引擎检测器（LLM 不可用时使用）
            empathy_engine: 规则引擎响应生成器（LLM 不可用时使用）
            llm_client: LLM 客户端（可选，启用 LLM 功能）
            enable_llm: 是否启用 LLM（当 llm_client 提供时生效）
        """
        # 记忆仓库（需要先创建以加载进化规则）
        self._memory = MemoryRepository(memory_path)
        evolved_rules = self._memory.load_evolved_rules()

        # 规则引擎组件
        self._rule_detector = detector or HumanEmotionDetector(evolved_rules=evolved_rules)
        self._rule_empathy = empathy_engine or HumanEmpathyEngine()

        # LLM 组件
        self._llm_client = llm_client
        self._enable_llm = enable_llm and LLM_AVAILABLE and llm_client is not None
        self._llm_detector: Optional[LLMEmotionDetector] = None
        self._llm_response_gen: Optional[LLMResponseGenerator] = None
        self._fallback_manager: Optional[FallbackManager] = None

        if self._enable_llm:
            self._llm_detector = LLMEmotionDetector(llm_client)
            self._llm_response_gen = LLMResponseGenerator(
                llm_client,
                fallback_engine=self._rule_empathy,
            )
            self._fallback_manager = FallbackManager()

        # 当前使用的检测器（兼容属性）
        self._detector = self._llm_detector if self._enable_llm else self._rule_detector
        self._empathy = self._llm_response_gen if self._enable_llm else self._rule_empathy

        # 其他组件
        self._irony = IronyDetector()
        self._context_analyzer = ContextualAnalyzer()
        self._conversation_contexts: Dict[str, ConversationContext] = {}

    @property
    def is_llm_enabled(self) -> bool:
        """是否启用了 LLM"""
        return self._enable_llm

    @property
    def is_llm_available(self) -> bool:
        """LLM 是否可用"""
        if not self._enable_llm:
            return False
        return self._llm_client is not None

    def analyze(self, text: str, options: Optional[AnalyzeOptions] = None) -> AnalysisResult:
        """
        分析文本情感

        Args:
            text: 输入文本
            options: 分析选项

        Returns:
            AnalysisResult: 完整分析结果
        """
        opts = options or AnalyzeOptions()

        # 1. 情感检测（优先使用 LLM，失败则降级到规则引擎）
        emotion_scores = self._detect_emotion(text)

        # 2. 获取主要情感和详细信息
        primary_emotion, primary_score = self._get_primary(emotion_scores)

        # LLM 可能返回更详细的 VAD 信息
        if self._enable_llm and hasattr(self, '_llm_detector') and self._llm_detector:
            try:
                llm_result = self._llm_detector.get_detailed_result(text)
                vad = (
                    llm_result.get("vad", {}).get("valence", 0.0),
                    llm_result.get("vad", {}).get("arousal", 0.0),
                    llm_result.get("vad", {}).get("dominance", 0.0),
                )
                explanation = llm_result.get("explanation")
                confidence = llm_result.get("confidence", primary_score)
            except Exception:
                vad = EMOTION_VAD.get(primary_emotion, (0.0, 0.0, 0.0))
                explanation = None
                confidence = primary_score
        else:
            vad = EMOTION_VAD.get(primary_emotion, (0.0, 0.0, 0.0))
            explanation = None
            confidence = primary_score

        intensity_label = get_intensity_label(primary_score)

        # 3. 反讽检测（规则引擎，因为 LLM 可能已理解语义）
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

        # 6. 生成共情回复（优先使用 LLM）
        human_response = self._generate_response(
            text=text,
            emotion=effective_emotion,
            intensity=effective_intensity,
            context=opts.context,
            user_id=opts.user_id,
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

        # 10. 获取检测解释（如果 LLM 没有返回）
        if explanation is None and primary_score > 0.1:
            explanation = self._rule_detector.explain(text) if hasattr(self._rule_detector, 'explain') else None

        # 构建explanation加入反讽信息
        if irony_result.is_irony:
            if explanation is None:
                explanation = {}
            if isinstance(explanation, dict):
                explanation["irony"] = {
                    "is_irony": True,
                    "surface_emotion": irony_result.surface_emotion,
                    "true_emotion": irony_result.true_emotion,
                    "confidence": irony_result.confidence,
                    "clues": irony_result.clues,
                }

        # 确定引擎版本
        engine_version = f"llm-v{__version__}" if self._enable_llm else f"rule-v{__version__}"

        return AnalysisResult(
            version=__version__,
            engine=engine_version,
            emotion=EmotionOutput(
                primary=effective_emotion,
                intensity=effective_intensity,
                vad=vad,
                confidence=confidence,
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
                empathy_depth=getattr(human_response, 'tone', '温暖'),
                tone=getattr(human_response, 'tone', '温暖'),
            ),
            user_profile=user_profile,
            context_used=opts.context is not None,
            emotion_mix=emotion_mix,
            explanation=explanation,
        )

    def _detect_emotion(self, text: str) -> Dict[str, float]:
        """
        情感检测（自动降级）

        优先使用 LLM，失败则降级到规则引擎
        """
        if not self._enable_llm or not self._llm_detector:
            return self._rule_detector.detect(text)

        try:
            # 尝试使用 LLM 检测
            return self._llm_detector.detect(text)
        except Exception as e:
            import logging
            logging.warning(f"LLM detection failed, falling back to rules: {e}")
            if self._fallback_manager:
                self._fallback_manager.record_failure(e)
            return self._rule_detector.detect(text)

    def _generate_response(
        self,
        text: str,
        emotion: str,
        intensity: float,
        context: Optional[str],
        user_id: str,
    ):
        """
        生成共情响应（自动降级）

        优先使用 LLM，失败则降级到规则引擎
        """
        if not self._enable_llm or not self._llm_response_gen:
            return self._rule_empathy.generate(
                emotion=emotion,
                intensity=intensity,
                context=context,
            )

        try:
            # 获取用户历史
            profile = self._memory.get_user(user_id)
            history = profile.emotional_history[-5:] if profile.emotional_history else []

            return self._llm_response_gen.generate(
                text=text,
                emotion=emotion,
                intensity=intensity,
                context={"context": context} if context else None,
                user_profile={
                    "relationship_level": profile.relationship_level,
                    "total_interactions": profile.total_interactions,
                },
                conversation_history=history,
            )
        except Exception as e:
            import logging
            logging.warning(f"LLM response generation failed, falling back to rules: {e}")
            if self._fallback_manager:
                self._fallback_manager.record_failure(e)
            return self._rule_empathy.generate(
                emotion=emotion,
                intensity=intensity,
                context=context,
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
            "gratitude_love", "hope_fear", "jealous_love",
            "frustration_hopelessness", "love_admiration",
            "anger_sadness", "shock_denial", "relief_sadness", "angry_fear",
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
            "confusion": "困惑", "nostalgia": "怀旧",
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
