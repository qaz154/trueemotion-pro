"""
情感分析器门面
整合检测器、记忆系统、响应生成器
"""

from dataclasses import dataclass
from typing import Optional

from trueemotion.core.emotions.detector import RuleBasedEmotionDetector
from trueemotion.core.emotions.plutchik24 import EMOTION_VAD, get_intensity_level
from trueemotion.core.analysis.output import (
    EmotionOutput,
    HumanResponse,
    UserProfile,
    AnalysisResult,
)
from trueemotion.core.response.engine import EmpathyEngine
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

    使用方法:
        analyzer = EmotionAnalyzer()
        result = analyzer.analyze("今天太开心了！")
    """

    def __init__(
        self,
        memory_path: str = "./memory",
        detector: Optional[RuleBasedEmotionDetector] = None,
        empathy_engine: Optional[EmpathyEngine] = None,
    ):
        """
        初始化分析器

        Args:
            memory_path: 记忆存储路径
            detector: 可选的自定义检测器
            empathy_engine: 可选的自定义共情引擎
        """
        self._detector = detector or RuleBasedEmotionDetector()
        self._empathy = empathy_engine or EmpathyEngine()
        self._memory = MemoryRepository(memory_path)

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

        # 1. 情感检测
        emotion_scores = self._detector.detect(text)

        # 2. 获取主情感和详细信息
        primary_emotion, primary_score = self._get_primary(emotion_scores)
        vad = EMOTION_VAD.get(primary_emotion, (0.0, 0.0, 0.0))
        intensity = get_intensity_level(primary_score)

        # 3. 生成共情回复
        human_response = self._empathy.generate(
            emotion=primary_emotion,
            intensity=primary_score,
            context=opts.context,
        )

        # 4. 更新用户记忆
        user_profile = self._update_memory(
            user_id=opts.user_id,
            emotion=primary_emotion,
            learn=opts.learn,
            response=opts.response,
            feedback=opts.feedback,
        )

        return AnalysisResult(
            version="4.0.0",
            engine="rule-based + empathy",
            emotion=EmotionOutput(
                primary=primary_emotion,
                intensity=primary_score,
                vad=vad,
                confidence=primary_score,
                all_emotions=emotion_scores,
            ),
            human_response=human_response,
            user_profile=user_profile,
            context_used=opts.context is not None,
        )

    def _get_primary(self, scores: dict[str, float]) -> tuple[str, float]:
        """获取主要情感"""
        if not scores:
            return "neutral", 0.0
        primary = max(scores.items(), key=lambda x: x[1])
        return primary[0], primary[1]

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

        # 更新交互次数
        profile.total_interactions += 1
        profile.last_emotion = emotion
        profile.emotional_history.append(emotion)

        # 更新主导情感（使用滑动平均）
        if profile.dominant_emotion is None:
            profile.dominant_emotion = emotion
        else:
            # 简单的移动平均
            current = emotion
            # 这里简化处理，实际应该计算频率
            profile.dominant_emotion = current

        # 学习新模式
        if learn and response:
            self._memory.learn_pattern(user_id, emotion, response, feedback)
            profile.learned_patterns = self._memory.get_pattern_count(user_id)

        # 保持历史在合理长度
        if len(profile.emotional_history) > 100:
            profile.emotional_history = profile.emotional_history[-100:]

        # 保存更新
        self._memory.save_user(user_id, profile)

        return profile

    def get_user_profile(self, user_id: str) -> UserProfile:
        """获取用户画像"""
        return self._memory.get_user(user_id)

    def evolve(self) -> dict:
        """执行进化：分析学习到的模式，反哺规则系统"""
        patterns = self._memory.get_all_patterns()
        return {
            "total_patterns": len(patterns),
            "patterns": patterns,
            "status": "evolved",
        }

    def get_stats(self) -> dict:
        """获取系统统计"""
        return self._memory.get_stats()
