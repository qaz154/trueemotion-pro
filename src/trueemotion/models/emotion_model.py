# -*- coding: utf-8 -*-
"""
TrueEmotion Model - 核心情感分析模型
====================================

多任务情感分析模型，同时预测：
1. 原型情感分类（8类）
2. 复合情感多标签（16类）
3. VAD维度回归（3个连续值）
4. 情感强度回归（1个连续值）

由于这是一个演示系统，我们使用基于规则和词典的方法，
但在真实场景中可以用深度学习模型替代
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trueemotion.emotion.plutchik24 import (
    EMOTION_DEFINITIONS, VAD_LEXICON, get_primary_emotions,
    get_complex_emotions, get_all_emotions
)
from trueemotion.emotion.emotion_output import EmotionOutput
from trueemotion.models.context_encoder import ContextEncoder, EmotionPatternMatcher
from trueemotion.models.irony_detector import IronyDetector


@dataclass
class EmotionModelConfig:
    """模型配置"""
    # 情感识别阈值
    primary_threshold: float = 0.3
    complex_threshold: float = 0.35

    # 置信度阈值
    confidence_threshold: float = 0.4

    # 上下文窗口大小
    context_window: int = 5

    # 是否启用反讽检测
    enable_irony: bool = True

    # 是否启用上下文
    enable_context: bool = True


class TrueEmotionModel:
    """
    TrueEmotion核心情感分析模型

    使用混合方法：
    1. 词典方法 - 基于VAD词典的快速匹配
    2. 规则方法 - 基于情感规则的组合
    3. 上下文方法 - 利用对话上下文
    4. 反讽检测 - 识别反讽表达

    在真实场景中，这些规则方法可以被深度学习模型替代
    """

    def __init__(self, config: Optional[EmotionModelConfig] = None):
        self.config = config or EmotionModelConfig()
        self.context_encoder = ContextEncoder(window_size=self.config.context_window)
        self.irony_detector = IronyDetector() if self.config.enable_irony else None

        # 初始化词典
        self._build_emotion_lexicon()

    def _build_emotion_lexicon(self) -> None:
        """构建情感词典"""
        # 为每种情感构建关键词列表和权重
        self.emotion_lexicon = {}
        self.emotion_patterns = {}

        for emotion_name, definition in EMOTION_DEFINITIONS.items():
            # 收集关键词
            keywords = list(definition.keywords)
            self.emotion_lexicon[emotion_name] = {
                "keywords": keywords,
                "vad": definition.vad,
                "primary_components": [p.value for p in definition.primary_components]
            }

    def analyze(self, text: str, context: Optional[List[str]] = None) -> EmotionOutput:
        """
        分析文本情感

        Args:
            text: 输入文本
            context: 可选的上下文文本列表

        Returns:
            EmotionOutput: 情感分析结果
        """
        text = text.strip()
        if not text:
            return EmotionOutput()

        # 1. 反讽检测
        is_irony = False
        irony_confidence = 0.0
        surface_emotion = None
        true_emotion = None

        if self.irony_detector and self.config.enable_irony:
            irony_result = self.irony_detector.detect(text, context)
            is_irony = irony_result.is_irony
            irony_confidence = irony_result.confidence
            if is_irony:
                surface_emotion = irony_result.surface_emotion
                true_emotion = irony_result.true_emotion

        # 2. 如果是反讽且置信度高，返回反讽情感
        if is_irony and irony_confidence > 0.6:
            output = self._create_irony_output(text, irony_result)
            return output

        # 3. 正常情感分析
        return self._analyze_normal(text, context)

    def _analyze_normal(self, text: str, context: Optional[List[str]] = None) -> EmotionOutput:
        """正常情感分析（非反讽）"""

        # 提取基础特征
        intensity_modifier, modifier_word = EmotionPatternMatcher.extract_intensity_modifier(text)
        has_negation = EmotionPatternMatcher.has_negation(text)

        # 1. 计算原型情感得分
        primary_scores = self._calculate_primary_scores(text)

        # 2. 计算复合情感
        complex_labels = self._calculate_complex_emotions(text, primary_scores)

        # 3. 计算VAD维度
        vad = self._calculate_vad(text, primary_scores)

        # 4. 计算情感强度
        intensity = self._calculate_intensity(text, primary_scores, intensity_modifier)

        # 5. 计算置信度
        confidence = self._calculate_confidence(primary_scores, text)

        # 6. 应用上下文调整
        if context and self.config.enable_context:
            vad = self._adjust_vad_with_context(vad, context)
            primary_scores = self._adjust_scores_with_context(primary_scores, context)

        # 7. 创建输出
        output = EmotionOutput(
            primary=primary_scores,
            complex=complex_labels,
            vad=vad,
            intensity=intensity,
            is_irony=False,
            irony_confidence=0.0,
            confidence=confidence,
            state="ACTIVE" if intensity > 0.5 else "BASELINE"
        )

        return output

    def _create_irony_output(self, text: str, irony_result) -> EmotionOutput:
        """创建反讽输出"""
        # 表面情感（用于回复）
        surface_primary = {"joy": 0.8}

        # 真实情感（用于理解）
        true_emotion = irony_result.true_emotion
        true_primary = {true_emotion: 0.9}

        # 复合情感
        complex_labels = {}
        if true_emotion in EMOTION_DEFINITIONS:
            definition = EMOTION_DEFINITIONS[true_emotion]
            for comp in definition.primary_components:
                complex_labels[f"{comp.value}_related"] = True

        return EmotionOutput(
            primary=true_primary,
            complex=complex_labels,
            vad=EMOTION_DEFINITIONS[true_emotion].vad,
            intensity=0.75,
            is_irony=True,
            irony_confidence=irony_result.confidence,
            surface_emotion=irony_result.surface_emotion,
            true_emotion=true_emotion,
            confidence=irony_result.confidence,
            state="ACTIVE"
        )

    def _calculate_primary_scores(self, text: str) -> Dict[str, float]:
        """计算原型情感得分"""
        text_lower = text.lower()
        scores = {}

        # 遍历每种情感
        for emotion_name, lexicon_info in self.emotion_lexicon.items():
            score = 0.0
            matched_keywords = []

            for keyword in lexicon_info["keywords"]:
                if keyword.lower() in text_lower:
                    score += 1.0
                    matched_keywords.append(keyword)

            if matched_keywords:
                # 归一化得分
                scores[emotion_name] = min(1.0, score / len(matched_keywords) * 0.5 + 0.3)
            else:
                scores[emotion_name] = 0.0

        # 如果没有任何匹配，使用VAD词典推断
        if max(scores.values()) < 0.1:
            for word, vad in VAD_LEXICON.items():
                if word in text_lower:
                    # 简单映射到最近的基础情感
                    if vad[0] > 0.5 and vad[1] > 0.3:
                        scores["joy"] = scores.get("joy", 0.0) + 0.4
                    elif vad[0] < -0.5 and vad[1] > 0.3:
                        scores["anger"] = scores.get("anger", 0.0) + 0.4
                    elif vad[0] < -0.5 and vad[1] < 0.0:
                        scores["sadness"] = scores.get("sadness", 0.0) + 0.4

        # 归一化到概率分布
        total = sum(scores.values())
        if total > 0:
            for emotion in scores:
                scores[emotion] /= total

        # 应用阈值
        for emotion in scores:
            if scores[emotion] < self.config.primary_threshold:
                scores[emotion] = 0.0

        return scores

    def _calculate_complex_emotions(self, text: str, primary_scores: Dict[str, float]) -> Dict[str, bool]:
        """计算复合情感"""
        complex_labels = {}

        # 获取高强度的原型情感
        active_primaries = {k: v for k, v in primary_scores.items() if v > 0.3}

        # 检查每种复合情感
        for emotion_name, definition in EMOTION_DEFINITIONS.items():
            if emotion_name in get_primary_emotions():
                continue

            # 获取组成该复合情感的原型
            required = set(p.value for p in definition.primary_components)
            present = set(active_primaries.keys())

            # 检查是否包含所有必需的原型情感
            overlap = required & present

            if len(overlap) >= len(required) * 0.6:  # 至少60%匹配
                # 计算置信度
                confidence = sum(primary_scores.get(p, 0) for p in overlap) / len(required)
                complex_labels[emotion_name] = confidence > self.config.complex_threshold

        return complex_labels

    def _calculate_vad(self, text: str, primary_scores: Dict[str, float]) -> Tuple[float, float, float]:
        """计算VAD维度"""
        # 方法1：从原型情感加权平均
        if primary_scores and sum(primary_scores.values()) > 0:
            total_weight = sum(primary_scores.values())
            v_sum, a_sum, d_sum = 0.0, 0.0, 0.0

            for emotion, score in primary_scores.items():
                if emotion in EMOTION_DEFINITIONS:
                    vad = EMOTION_DEFINITIONS[emotion].vad
                    weight = score / total_weight
                    v_sum += vad[0] * weight
                    a_sum += vad[1] * weight
                    d_sum += vad[2] * weight

            return (v_sum, a_sum, d_sum)

        # 方法2：从词典推断
        text_lower = text.lower()
        v_sum, a_sum, d_sum = 0.0, 0.0, 0.0
        count = 0

        for word, vad in VAD_LEXICON.items():
            if word in text_lower:
                v_sum += vad[0]
                a_sum += vad[1]
                d_sum += vad[2]
                count += 1

        if count > 0:
            return (v_sum / count, a_sum / count, d_sum / count)

        return (0.0, 0.0, 0.0)

    def _calculate_intensity(self, text: str, primary_scores: Dict[str, float],
                            modifier: float = 1.0) -> float:
        """计算情感强度"""
        # 基础强度
        if primary_scores:
            max_score = max(primary_scores.values())
            base_intensity = max_score
        else:
            base_intensity = 0.3

        # 强度修饰词调整
        intensity = base_intensity * modifier

        # 标点符号增强
        exclamation_count = text.count("!") + text.count("！")
        question_count = text.count("?") + text.count("？")

        if exclamation_count > 0:
            intensity = min(1.0, intensity + exclamation_count * 0.1)
        if question_count > 0:
            intensity = min(1.0, intensity + question_count * 0.05)

        # 重复字符增强（如"太！！"）
        repeat_pattern = re.findall(r'(.)\1{2,}', text)
        if repeat_pattern:
            intensity = min(1.0, intensity + 0.1)

        return min(1.0, max(0.0, intensity))

    def _calculate_confidence(self, primary_scores: Dict[str, float], text: str) -> float:
        """计算模型置信度"""
        # 基于得分分布的置信度
        if not primary_scores or sum(primary_scores.values()) == 0:
            return 0.3

        scores = list(primary_scores.values())
        max_score = max(scores)
        second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0

        # 如果最高分远高于第二名，置信度高
        gap = max_score - second_max
        if gap > 0.5:
            confidence = 0.8 + gap * 0.2
        elif gap > 0.2:
            confidence = 0.6 + gap * 0.5
        else:
            confidence = 0.5

        # 基于关键词匹配的置信度
        matched_keywords = sum(1 for s in primary_scores.values() if s > 0)
        if matched_keywords >= 3:
            confidence = min(0.95, confidence + 0.1)

        return min(0.95, max(0.3, confidence))

    def _adjust_vad_with_context(self, vad: Tuple[float, float, float],
                                  context: List[str]) -> Tuple[float, float, float]:
        """利用上下文调整VAD"""
        if not context:
            return vad

        v, a, d = vad

        # 检查上下文情感趋势
        positive_count = sum(1 for c in context[-3:] if any(
            word in c.lower() for word in ["开心", "高兴", "棒", "好", "喜欢"]
        ))
        negative_count = sum(1 for c in context[-3:] if any(
            word in c.lower() for word in ["难过", "生气", "害怕", "担心", "累"]
        ))

        # 上下文调整
        if positive_count > negative_count * 2:
            v = min(1.0, v + 0.1)  # 偏正面
        elif negative_count > positive_count * 2:
            v = max(-1.0, v - 0.1)  # 偏负面

        return (v, a, d)

    def _adjust_scores_with_context(self, scores: Dict[str, float],
                                     context: List[str]) -> Dict[str, float]:
        """利用上下文调整情感得分"""
        if not context:
            return scores

        # 检查上下文中的情感
        context_text = " ".join(context[-3:]).lower()

        for emotion in get_primary_emotions():
            if emotion in EMOTION_DEFINITIONS:
                keywords = EMOTION_DEFINITIONS[emotion].keywords
                if any(kw.lower() in context_text for kw in keywords):
                    # 上下文中有这种情感，增强当前得分
                    if emotion in scores:
                        scores[emotion] = min(1.0, scores[emotion] * 1.2)

        return scores

    def reset_context(self) -> None:
        """重置上下文"""
        self.context_encoder.reset()


class EmotionAnalyzer:
    """
    情感分析器 - 对外接口

    封装TrueEmotionModel，提供简洁的API
    """

    def __init__(self):
        self.model = TrueEmotionModel()
        self.context: List[str] = []

    def analyze(self, text: str, reset_context: bool = False) -> EmotionOutput:
        """
        分析文本情感

        Args:
            text: 输入文本
            reset_context: 是否重置上下文

        Returns:
            EmotionOutput: 情感分析结果
        """
        if reset_context:
            self.context = []

        # 分析情感
        result = self.model.analyze(text, self.context)

        # 更新上下文
        self.context.append(text)
        if len(self.context) > 10:
            self.context.pop(0)

        return result

    def analyze_batch(self, texts: List[str]) -> List[EmotionOutput]:
        """批量分析"""
        return [self.analyze(text) for text in texts]


if __name__ == "__main__":
    # 测试情感分析器
    analyzer = EmotionAnalyzer()

    test_texts = [
        # 原始情感
        "今天太开心了！终于完成了项目！",
        "我很难过，失恋了...",
        "真是气死我了，又被骗了！",
        "好害怕啊，担心明天的考试...",
        "好期待！下个月就要去旅游了！",

        # 复合情感
        "看着他成功了，既高兴又有点嫉妒。",
        "又失望又生气，真是受够了！",
        "又惊又喜，简直不敢相信！",

        # 反讽
        "还行吧，就那样。",
        "真是太感谢了，让我等了三个小时。",
        "太好了，又迟到了。",
        "好怕怕哦，吓死人了呢。",

        # 真正面
        "太开心了！终于成功了！",
        "谢谢你的礼物，我很喜欢！",
    ]

    print("=" * 70)
    print("TrueEmotion 情感分析测试")
    print("=" * 70)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\n文本: {text}")
        print(f"结果: {result}")
        print(f"  原型: {result.primary}")
        print(f"  复合: {[k for k, v in result.complex.items() if v]}")
        print(f"  VAD: V={result.vad[0]:.2f}, A={result.vad[1]:.2f}, D={result.vad[2]:.2f}")
        print(f"  强度: {result.intensity:.2f}")
        print(f"  反讽: {result.is_irony} (置信度: {result.irony_confidence:.2f})")
        if result.is_irony:
            print(f"    表面: {result.surface_emotion} -> 真实: {result.true_emotion}")
