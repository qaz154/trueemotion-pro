# -*- coding: utf-8 -*-
"""
TrueEmotion 完整集成系统
========================

集成了所有模块的完整情感AI系统：
1. 24种情感识别（Plutchik模型）
2. 情感强度预测
3. VAD维度表示
4. 反讽检测
5. 上下文理解
6. 自动进化学习

使用示例：
    system = TrueEmotionSystem()
    result = system.analyze("今天太开心了！")
    print(result)
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trueemotion.emotion.plutchik24 import (
    EMOTION_DEFINITIONS, VAD_LEXICON,
    get_primary_emotions, get_all_emotions
)
from trueemotion.emotion.emotion_output import EmotionOutput, EmotionContext
from trueemotion.models.context_encoder import ContextEncoder, EmotionPatternMatcher
from trueemotion.models.irony_detector import IronyDetector, IronyResult
from trueemotion.evolution.emotion_evolution import (
    EmotionEvolution, EvolutionManager,
    EbbinghausForgetting, LearnedRule
)


# ==================== 配置 ====================

@dataclass
class SystemConfig:
    """系统配置"""
    # 情感识别
    primary_threshold: float = 0.3
    complex_threshold: float = 0.35

    # 上下文
    context_window: int = 5

    # 进化
    enable_evolution: bool = True
    evolve_interval: int = 10  # 每N次交互进化一次
    pattern_min_count: int = 3  # 形成准则的最少重复次数

    # 反讽
    irony_threshold: float = 0.6


# ==================== 响应策略 ====================

RESPONSE_STRATEGIES = {
    # 用户情感 -> AI响应情感
    "joy": {
        "primary": "celebrate",
        "secondary": "share_joy",
        "examples": ["太棒了！", "恭喜恭喜！", "说说细节！"],
        "avoid": ["dismiss", "negative"]
    },
    "sadness": {
        "primary": "empathy",
        "secondary": "comfort",
        "examples": ["我懂你的感受...", "心疼你...", "慢慢来"],
        "avoid": ["dismiss", "advice", "positive_reframe"]
    },
    "anger": {
        "primary": "calm",
        "secondary": "understand",
        "examples": ["确实很气人...", "换我也生气...", "说说怎么回事"],
        "avoid": ["dismiss", "minimize", "criticize"]
    },
    "fear": {
        "primary": "reassure",
        "secondary": "support",
        "examples": ["别太担心...", "一步一步来...", "我陪着你"],
        "avoid": ["dismiss", "minimize", "rush"]
    },
    "surprise": {
        "primary": "acknowledge",
        "secondary": "curiosity",
        "examples": ["哇！真的假的！", "太意外了！", "什么！"],
        "avoid": ["dismiss", "understate"]
    },
    "anticipation": {
        "primary": "encourage",
        "secondary": "share_excitement",
        "examples": ["听起来很让人期待！", "加油！", "好羡慕啊！"],
        "avoid": ["dismiss", "negative", "kill_anticipation"]
    },
    "trust": {
        "primary": "appreciate",
        "secondary": "reciprocate",
        "examples": ["谢谢你的信任", "我会的", "一起努力"],
        "avoid": ["dismiss", "take_for_granted"]
    },
    "disgust": {
        "primary": "understand",
        "secondary": "validate",
        "examples": ["确实换谁都会不舒服", "理解你的感受", "需要发泄一下吗"],
        "avoid": ["dismiss", "minimize", "criticize"]
    },
    "anxiety": {
        "primary": "reassure",
        "secondary": "break_down",
        "examples": ["别着急", "先冷静下来", "我们一起想办法"],
        "avoid": ["rush", "dismiss", "underestimate"]
    },
    "envy": {
        "primary": "acknowledge",
        "secondary": "normalize",
        "examples": ["羡慕是正常的", "努力你也行的", "加油"],
        "avoid": ["dismiss", "minimize", "boast"]
    },
    "guilt": {
        "primary": "forgive",
        "secondary": "reassure",
        "examples": ["谁都会犯错", "别太自责", "下次会更好"],
        "avoid": ["criticize", "dismiss", "rub_it_in"]
    },
    "disappointment": {
        "primary": "empathy",
        "secondary": "reframe",
        "examples": ["确实很失望", "我能理解", "想想怎么改进"],
        "avoid": ["dismiss", "minimize", "blame"]
    },
    "remorse": {
        "primary": "forgive",
        "secondary": "support",
        "examples": ["过去就过去了", "重要的是现在", "我陪着你"],
        "avoid": ["criticize", "blame", "rub_it_in"]
    },
    "pride": {
        "primary": "celebrate",
        "secondary": "acknowledge",
        "examples": ["太骄傲了！", "真的很厉害！", "说说怎么做到的"],
        "avoid": ["dismiss", "undermine", "deflect"]
    },
    "contempt": {
        "primary": "defuse",
        "secondary": "understand",
        "examples": ["别太往心里去", "换个角度看", "重要的是你自己"],
        "avoid": ["argue", "defend", "escalate"]
    },
    "cynicism": {
        "primary": "understand",
        "secondary": "gentle_challenge",
        "examples": ["能理解你的感受", "也许没那么糟", "要不要试试"],
        "avoid": ["argue", "dismiss", "lecture"]
    },
    "neutral": {
        "primary": "listen",
        "secondary": "engage",
        "examples": ["嗯，了解了", "我知道了", "然后呢"],
        "avoid": []
    }
}


# ==================== 完整情感系统 ====================

class TrueEmotionSystem:
    """
    TrueEmotion 完整情感AI系统

    集成功能：
    - 多标签情感识别
    - 情感强度预测
    - VAD维度
    - 反讽检测
    - 上下文理解
    - 自动进化学习
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # 核心组件
        self.context_encoder = ContextEncoder(window_size=self.config.context_window)
        self.irony_detector = IronyDetector()
        self.evolution_manager = EvolutionManager() if self.config.enable_evolution else None

        # 情感词典
        self._build_emotion_lexicon()

        # 统计
        self.stats = {
            "total_analyzed": 0,
            "irony_detected": 0,
            "context_adjusted": 0,
            "evolution_triggered": 0
        }

    def _build_emotion_lexicon(self) -> None:
        """构建情感词典"""
        self.emotion_lexicon = {}
        for emotion_name, definition in EMOTION_DEFINITIONS.items():
            self.emotion_lexicon[emotion_name] = {
                "keywords": list(definition.keywords),
                "vad": definition.vad,
                "primary_components": [p.value for p in definition.primary_components]
            }

    # ==================== 核心接口 ====================

    def analyze(self, text: str, context: Optional[List[str]] = None,
                learn: bool = False, response: Optional[str] = None,
                feedback: Optional[float] = None) -> EmotionOutput:
        """
        分析文本情感

        Args:
            text: 输入文本
            context: 可选的上下文
            learn: 是否记录学习
            response: AI回复（用于学习）
            feedback: 用户反馈（用于学习）

        Returns:
            EmotionOutput: 情感分析结果
        """
        text = text.strip()
        if not text:
            return EmotionOutput()

        self.stats["total_analyzed"] += 1

        # 1. 反讽检测
        irony_result = self._detect_irony(text, context)

        # 2. 正常情感分析
        if irony_result.is_irony:
            output = self._create_irony_output(irony_result)
            self.stats["irony_detected"] += 1
        else:
            output = self._analyze_normal(text, context)

        # 3. 学习（如果启用）
        if learn and self.evolution_manager and response:
            self._learn_from_interaction(text, output, response, feedback)

        # 4. 更新上下文
        self.context_encoder.add_utterance(
            text=text,
            speaker="user",
            emotion=output.get_primary_emotion(),
            intensity=output.intensity
        )

        return output

    def generate_response(self, emotion_output: EmotionOutput,
                         intensity: Optional[str] = None) -> str:
        """
        基于情感生成响应

        Args:
            emotion_output: 情感分析结果
            intensity: 可选强度修饰（"high", "medium", "low"）

        Returns:
            str: 建议的响应
        """
        primary = emotion_output.get_primary_emotion() or "neutral"
        strategy = RESPONSE_STRATEGIES.get(primary, RESPONSE_STRATEGIES["neutral"])

        # 根据强度调整
        examples = strategy["examples"]
        if intensity == "high" and len(examples) > 1:
            examples = examples[1:]  # 更强烈的表达
        elif intensity == "low":
            examples = examples[:1]  # 更克制的表达

        return examples[0] if examples else "嗯，了解了。"

    def evolve(self) -> Dict[str, Any]:
        """
        执行进化迭代

        Returns:
            进化报告
        """
        if not self.evolution_manager:
            return {"error": "Evolution disabled"}

        report = self.evolution_manager.evolution.evolve()
        self.stats["evolution_triggered"] += 1

        return report

    # ==================== 内部方法 ====================

    def _detect_irony(self, text: str, context: Optional[List[str]]) -> IronyResult:
        """检测反讽"""
        # 获取上下文情感
        ctx_emotions = None
        if context:
            # 从context列表中提取情感
            ctx_emotions = []
            for ctx_text in context[-3:]:
                result = self._analyze_normal(ctx_text, None)
                if result.get_primary_emotion():
                    ctx_emotions.append(result.get_primary_emotion())

        return self.irony_detector.detect(text, ctx_emotions)

    def _analyze_normal(self, text: str, context: Optional[List[str]]) -> EmotionOutput:
        """正常情感分析"""
        # 提取特征
        intensity_modifier, _ = EmotionPatternMatcher.extract_intensity_modifier(text)

        # 计算情感得分
        primary_scores = self._calculate_primary_scores(text)

        # 计算复合情感
        complex_labels = self._calculate_complex_emotions(text, primary_scores)

        # 计算VAD
        vad = self._calculate_vad(text, primary_scores)

        # 计算强度
        intensity = self._calculate_intensity(text, primary_scores, intensity_modifier)

        # 计算置信度
        confidence = self._calculate_confidence(primary_scores, text)

        # 上下文调整
        if context:
            vad = self._adjust_with_context(vad, context)
            self.stats["context_adjusted"] += 1

        return EmotionOutput(
            primary=primary_scores,
            complex=complex_labels,
            vad=vad,
            intensity=intensity,
            confidence=confidence,
            state="ACTIVE" if intensity > 0.5 else "BASELINE"
        )

    def _create_irony_output(self, irony_result: IronyResult) -> EmotionOutput:
        """创建反讽输出"""
        true_emotion = irony_result.true_emotion
        definition = EMOTION_DEFINITIONS.get(true_emotion)

        return EmotionOutput(
            primary={true_emotion: 0.9},
            complex={},
            vad=definition.vad if definition else (0.0, 0.0, 0.0),
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

        for emotion_name, lexicon_info in self.emotion_lexicon.items():
            score = 0.0
            matched = []

            for keyword in lexicon_info["keywords"]:
                if keyword.lower() in text_lower:
                    score += 1.0
                    matched.append(keyword)

            if matched:
                scores[emotion_name] = min(1.0, score / len(matched) * 0.5 + 0.3)
            else:
                scores[emotion_name] = 0.0

        # 如果没有匹配，使用VAD词典
        if max(scores.values()) < 0.1:
            for word, vad in VAD_LEXICON.items():
                if word in text_lower:
                    if vad[0] > 0.5:
                        scores["joy"] = scores.get("joy", 0.0) + 0.4
                    elif vad[0] < -0.5 and vad[1] > 0.3:
                        scores["anger"] = scores.get("anger", 0.0) + 0.4
                    elif vad[0] < -0.5 and vad[1] < 0.0:
                        scores["sadness"] = scores.get("sadness", 0.0) + 0.4

        # 归一化
        total = sum(scores.values())
        if total > 0:
            for e in scores:
                scores[e] /= total

        # 应用阈值
        for e in scores:
            if scores[e] < self.config.primary_threshold:
                scores[e] = 0.0

        return scores

    def _calculate_complex_emotions(self, text: str,
                                     primary_scores: Dict[str, float]) -> Dict[str, bool]:
        """计算复合情感"""
        complex_labels = {}
        active = {k: v for k, v in primary_scores.items() if v > 0.3}

        for emotion_name, definition in EMOTION_DEFINITIONS.items():
            if emotion_name in get_primary_emotions():
                continue

            required = set(p.value for p in definition.primary_components)
            present = set(active.keys())
            overlap = required & present

            if len(overlap) >= len(required) * 0.6:
                confidence = sum(primary_scores.get(p, 0) for p in overlap) / len(required)
                complex_labels[emotion_name] = confidence > self.config.complex_threshold

        return complex_labels

    def _calculate_vad(self, text: str, primary_scores: Dict[str, float]) -> Tuple[float, float, float]:
        """计算VAD"""
        if primary_scores and sum(primary_scores.values()) > 0:
            total = sum(primary_scores.values())
            v, a, d = 0.0, 0.0, 0.0

            for emotion, score in primary_scores.items():
                if emotion in EMOTION_DEFINITIONS:
                    vad = EMOTION_DEFINITIONS[emotion].vad
                    w = score / total
                    v += vad[0] * w
                    a += vad[1] * w
                    d += vad[2] * w

            return (v, a, d)

        text_lower = text.lower()
        v, a, d, count = 0.0, 0.0, 0.0, 0

        for word, vad in VAD_LEXICON.items():
            if word in text_lower:
                v += vad[0]
                a += vad[1]
                d += vad[2]
                count += 1

        if count > 0:
            return (v / count, a / count, d / count)

        return (0.0, 0.0, 0.0)

    def _calculate_intensity(self, text: str, primary_scores: Dict[str, float],
                              modifier: float = 1.0) -> float:
        """计算强度"""
        base = max(primary_scores.values()) if primary_scores else 0.3
        intensity = base * modifier

        # 标点增强
        exclamation = text.count("!") + text.count("！")
        question = text.count("?") + text.count("？")

        if exclamation > 0:
            intensity = min(1.0, intensity + exclamation * 0.1)
        if question > 0:
            intensity = min(1.0, intensity + question * 0.05)

        return min(1.0, max(0.0, intensity))

    def _calculate_confidence(self, primary_scores: Dict[str, float], text: str) -> float:
        """计算置信度"""
        if not primary_scores or sum(primary_scores.values()) == 0:
            return 0.3

        scores = sorted(primary_scores.values(), reverse=True)
        max_score = scores[0]
        second = scores[1] if len(scores) > 1 else 0

        gap = max_score - second
        confidence = 0.5 + gap * 0.5 if gap > 0.2 else 0.5

        matched = sum(1 for s in primary_scores.values() if s > 0)
        if matched >= 3:
            confidence = min(0.95, confidence + 0.1)

        return min(0.95, max(0.3, confidence))

    def _adjust_with_context(self, vad: Tuple[float, float, float],
                              context: List[str]) -> Tuple[float, float, float]:
        """上下文调整"""
        v, a, d = vad

        # 简单的上下文情感检测
        positive = sum(1 for t in context[-3:] if any(
            w in t.lower() for w in ["开心", "高兴", "棒", "好", "喜欢"]
        ))
        negative = sum(1 for t in context[-3:] if any(
            w in t.lower() for w in ["难过", "生气", "害怕", "担心", "累"]
        ))

        if positive > negative * 2:
            v = min(1.0, v + 0.1)
        elif negative > positive * 2:
            v = max(-1.0, v - 0.1)

        return (v, a, d)

    def _learn_from_interaction(self, text: str, output: EmotionOutput,
                                 response: str, feedback: Optional[float]) -> None:
        """从交互中学习"""
        if not self.evolution_manager:
            return

        # 获取响应情感
        response_emotion = self._infer_response_emotion(response)

        # 记录交互
        self.evolution_manager.process_interaction(
            user_text=text,
            user_emotion=output.get_primary_emotion() or "neutral",
            user_intensity=output.intensity,
            response=response,
            response_emotion=response_emotion,
            feedback=feedback
        )

    def _infer_response_emotion(self, response: str) -> str:
        """推断响应情感"""
        response_lower = response.lower()

        # 简单关键词匹配
        if any(w in response_lower for w in ["心疼", "理解", "难过"]):
            return "empathy"
        if any(w in response_lower for w in ["棒", "厉害", "恭喜"]):
            return "celebrate"
        if any(w in response_lower for w in ["加油", "相信"]):
            return "encourage"
        if any(w in response_lower for w in ["别担心", "没事"]):
            return "reassure"
        if any(w in response_lower for w in ["确实", "换我"]):
            return "understand"

        return "neutral"

    # ==================== 工具方法 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        stats = dict(self.stats)
        if self.evolution_manager:
            stats["evolution"] = self.evolution_manager.evolution.get_stats()
        return stats

    def get_rules(self) -> List[Dict]:
        """获取当前准则"""
        if self.evolution_manager:
            return self.evolution_manager.evolution.get_rules()
        return []

    def reset_context(self) -> None:
        """重置上下文"""
        self.context_encoder.reset()

    def reset_evolution(self) -> None:
        """重置进化状态"""
        if self.evolution_manager:
            self.evolution_manager.evolution.reset()


# ==================== 便捷函数 ====================

_system_instance: Optional[TrueEmotionSystem] = None


def get_system() -> TrueEmotionSystem:
    """获取系统单例"""
    global _system_instance
    if _system_instance is None:
        _system_instance = TrueEmotionSystem()
    return _system_instance


def analyze(text: str, **kwargs) -> EmotionOutput:
    """便捷分析函数"""
    return get_system().analyze(text, **kwargs)


def analyze_and_respond(text: str, **kwargs) -> Tuple[EmotionOutput, str]:
    """分析并生成响应"""
    system = get_system()
    emotion = system.analyze(text, **kwargs)
    response = system.generate_response(emotion)
    return emotion, response


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("TrueEmotion 完整集成系统测试")
    print("=" * 70)

    system = TrueEmotionSystem()

    # 测试情感分析
    print("\n【情感分析测试】")
    test_texts = [
        "今天太开心了！终于完成了项目！",
        "我很难过，失恋了...",
        "真是气死我了，又被骗了！",
        "好害怕啊，担心明天的考试...",
        "好期待！下个月就要去旅游了！",
        "看着他成功了，既高兴又有点嫉妒。",
        "还行吧，就那样。",
        "真是太感谢了，让我等了三个小时。",
        "太好了，又迟到了。",
    ]

    for text in test_texts:
        result = system.analyze(text)
        irony = " [反讽]" if result.is_irony else ""
        print(f"  \"{text[:20]}...\"")
        print(f"    -> {result.get_primary_emotion():10s} 强度:{result.intensity:.2f}{irony}")

    # 测试带学习的对话
    print("\n【学习测试】")
    dialogues = [
        ("工作好累啊...", "辛苦了，要注意休息。", 0.8),
        ("项目又延期了，好烦", "确实挺烦的，说说怎么回事", 0.9),
        ("太棒了！终于完成了！", "恭喜恭喜！太厉害了！", 0.9),
        ("担心明天的考试...", "别太紧张，相信自己可以的", 0.8),
    ]

    for user_text, response, feedback in dialogues:
        result = system.analyze(
            user_text,
            learn=True,
            response=response,
            feedback=feedback
        )
        fb = f"+{feedback}" if feedback else ""
        print(f"  [{result.get_primary_emotion()}] {user_text[:15]}... -> {fb}")

    # 执行进化
    print("\n【执行进化】")
    report = system.evolve()
    print(f"  总交互: {report.get('total_interactions', 0)}")
    print(f"  新准则: {report.get('new_rules', [])}")
    print(f"  反思: {report.get('reflections', [])}")

    # 生成响应
    print("\n【响应生成测试】")
    emotions_to_test = ["joy", "sadness", "anger", "fear", "anticipation"]

    for emotion in emotions_to_test:
        mock_output = EmotionOutput(
            primary={emotion: 0.8},
            intensity=0.8,
            state="ACTIVE"
        )
        response = system.generate_response(mock_output)
        print(f"  {emotion:10s} -> \"{response}\"")

    # 统计
    print("\n【系统统计】")
    stats = system.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
