"""
情感分析器门面
====================
整合检测器、记忆系统、响应生成器
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict
import logging

from trueemotion._version import __version__

from trueemotion.core.emotions.detector import HumanEmotionDetector
from trueemotion.core.emotions.irony import IronyDetector
from trueemotion.core.emotions.i18n import EMOTION_CN
from trueemotion.core.analysis.context import ContextualAnalyzer
from trueemotion.core.analysis.output import (
    EmotionOutput,
    HumanResponse,
    AnalysisResult,
)
from trueemotion.core.analysis.pipeline import EmotionPipeline
from trueemotion.core.analysis.response_builder import ResponseBuilder
from trueemotion.core.response.engine import HumanEmpathyEngine
from trueemotion.memory.repository import MemoryRepository, UserProfile

# LLM 组件（可选）
try:
    from trueemotion.core.llm import (
        BaseLLMClient,
        OpenAIClient,
        LLMEmotionDetector,
        LLMResponseGenerator,
        FallbackManager,
    )
    from trueemotion.core.llm.base import LLMError
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLMClient = None
    OpenAIClient = None
    LLMEmotionDetector = None
    LLMResponseGenerator = None
    FallbackManager = None
    LLMError = Exception


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

    功能:
    - analyze() 精简为 ~30 行编排器
    - EmotionPipeline 负责检测 / VAD / 反讽 / 上下文
    - ResponseBuilder 负责共情响应 + 追问
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
        memory_repo: Optional[MemoryRepository] = None,
    ):
        self._memory = memory_repo or MemoryRepository(memory_path)
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
        self._conversation_contexts: Dict[str, object] = {}

        # Pipeline + ResponseBuilder (initialised once)
        self._pipeline = EmotionPipeline(
            rule_detector=self._rule_detector,
            irony_detector=self._irony,
            context_analyzer=self._context_analyzer,
            llm_detector=self._llm_detector,
            fallback_manager=self._fallback_manager,
            enable_llm=self._enable_llm,
        )
        self._response_builder = ResponseBuilder(
            rule_empathy=self._rule_empathy,
            context_analyzer=self._context_analyzer,
            memory=self._memory,
            llm_response_gen=self._llm_response_gen,
            fallback_manager=self._fallback_manager,
            enable_llm=self._enable_llm,
        )

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, text: str, options: Optional[AnalyzeOptions] = None) -> AnalysisResult:
        """分析文本情感 (orchestrator)."""
        opts = options or AnalyzeOptions()

        # Steps 1-5: detection, VAD, irony, context
        pipe = self._pipeline.run(text)

        # Steps 6-7: empathy response + follow-up
        human_response = self._response_builder.build(
            text=text,
            emotion=pipe.effective_emotion,
            intensity=pipe.effective_intensity,
            context=opts.context,
            user_id=opts.user_id,
            context_result=pipe.context_result,
        )

        # Step 8: emotion mix description
        emotion_mix = self._build_emotion_mix(pipe.emotion_scores)

        # Step 9: update user memory
        user_profile = self._update_memory(
            user_id=opts.user_id,
            emotion=pipe.effective_emotion,
            learn=opts.learn,
            response=opts.response,
            feedback=opts.feedback,
        )

        # Step 10: enrich explanation
        explanation = self._enrich_explanation(pipe)

        engine_version = f"llm-v{__version__}" if self._enable_llm else f"rule-v{__version__}"

        return AnalysisResult(
            version=__version__,
            engine=engine_version,
            emotion=EmotionOutput(
                primary=pipe.effective_emotion,
                intensity=pipe.effective_intensity,
                vad=pipe.vad,
                confidence=pipe.confidence,
                intensity_label=pipe.intensity_label,
                all_emotions=pipe.emotion_scores,
                compound_emotions=pipe.compound_emotions,
                emotion_mix=self._build_top_emotions_list(pipe.emotion_scores),
            ),
            human_response=human_response,
            user_profile=user_profile,
            context_used=opts.context is not None,
            emotion_mix=emotion_mix,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_explanation(self, pipe) -> Optional[Dict]:
        """Merge rule explanation and irony info into final explanation."""
        explanation = pipe.explanation

        # Fill in rule-based explanation when LLM didn't provide one
        if explanation is None and pipe.primary_score > 0.1:
            if hasattr(self._rule_detector, "explain"):
                explanation = self._rule_detector.explain(pipe.primary_emotion)

        # Attach irony info
        if pipe.irony_result.is_irony:
            if explanation is None:
                explanation = {}
            if isinstance(explanation, dict):
                explanation["irony"] = {
                    "is_irony": True,
                    "surface_emotion": pipe.irony_result.surface_emotion,
                    "true_emotion": pipe.irony_result.true_emotion,
                    "confidence": pipe.irony_result.confidence,
                    "clues": pipe.irony_result.clues,
                }

        return explanation

    def _build_emotion_mix(self, scores: dict) -> str:
        """构建情感混合描述"""
        if not scores:
            return "中性"

        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_emotions) == 1:
            return f"以{self._get_emotion_cn(sorted_emotions[0][0])}为主"

        if len(sorted_emotions) == 2:
            e1, _ = sorted_emotions[0]
            e2, _ = sorted_emotions[1]
            return f"以{self._get_emotion_cn(e1)}为主，伴有{self._get_emotion_cn(e2)}"

        e1, _ = sorted_emotions[0]
        e2, _ = sorted_emotions[1]
        e3, _ = sorted_emotions[2]
        return f"以{self._get_emotion_cn(e1)}为主，伴有{self._get_emotion_cn(e2)}和{self._get_emotion_cn(e3)}"

    @staticmethod
    def _build_top_emotions_list(scores: dict) -> list:
        """构建 Top 情感列表"""
        return [
            (e, round(s, 3))
            for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    @staticmethod
    def _get_emotion_cn(emotion: str) -> str:
        """获取情感中文名"""
        return EMOTION_CN.get(emotion, emotion)

    def _update_memory(
        self,
        user_id: str,
        emotion: str,
        learn: bool,
        response: Optional[str],
        feedback: float,
    ) -> UserProfile:
        """更新用户记忆 (immutable via dataclasses.replace)."""
        profile = self._memory.get_user(user_id)

        # Build updated history
        new_history = list(profile.emotional_history) + [emotion]
        if len(new_history) > 100:
            new_history = new_history[-100:]

        # Determine learned_patterns count
        learned_patterns = profile.learned_patterns
        if learn and response:
            self._memory.learn_pattern(user_id, emotion, response, feedback)
            learned_patterns = self._memory.get_pattern_count(user_id)

        profile = dataclasses.replace(
            profile,
            total_interactions=profile.total_interactions + 1,
            last_emotion=emotion,
            emotional_history=new_history,
            dominant_emotion=profile.dominant_emotion or emotion,
            learned_patterns=learned_patterns,
        )

        self._memory.save_user(user_id, profile)
        return profile

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> UserProfile:
        """获取用户画像"""
        return self._memory.get_user(user_id)

    def evolve(self) -> dict:
        """
        执行进化分析

        注意: 实际进化由 TrueEmotionPro.evolve() 通过 EvolutionManager 执行
        此方法仅返回当前模式统计，供内部使用
        """
        patterns = self._memory.get_all_patterns()
        total = sum(len(v) for v in patterns.values()) if isinstance(patterns, dict) else len(patterns)
        return {
            "total_patterns": total,
            "emotions_with_patterns": list(patterns.keys()) if isinstance(patterns, dict) else [],
            "evolved_rules": 0,
            "status": "use_trueemotion_pro_evolve",
        }

    def get_stats(self) -> dict:
        """获取系统统计"""
        return self._memory.get_stats()
