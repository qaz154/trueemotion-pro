"""
反讽检测器 v1.13
================
识别真实情感 vs 表面情感

核心理念:
1. 表面情感 - 文字直接表达的情感
2. 真实情感 - 实际想要表达的情感
3. 反讽模式 - "挺好的"可能是讽刺

反讽识别线索:
- 语气词和标点异常组合
- 矛盾修饰词
- 与情境不符的情感表达
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class IronyResult:
    """反讽检测结果"""
    is_irony: bool
    surface_emotion: str
    true_emotion: Optional[str]
    confidence: float  # 0.0-1.0
    clues: list[str]  # 检测到的线索


class IronyDetector:
    """
    反讽检测器

    检测常见的反讽模式，如:
    - "挺好的" (实际上可能是不满)
    - "你可真是太好了" (可能是讽刺)
    - "哇，好厉害啊" (可能是讽刺)
    """

    # 反讽模式及其真实情感
    IRONY_PATTERNS = {
        # 表面正面，实际负面
        ("positive", "挺好", "disgust"): 0.8,
        ("positive", "真棒", "contempt"): 0.7,
        ("positive", "厉害", "envy"): 0.6,
        ("positive", "优秀", "contempt"): 0.7,
        ("positive", "太好了", "anger"): 0.6,
        ("positive", "谢谢啊", "anger"): 0.7,
        ("positive", "感动", "disgust"): 0.5,
        ("positive", "太棒了", "contempt"): 0.7,
        ("positive", "绝了", "disgust"): 0.6,
        ("positive", "完美", "contempt"): 0.6,

        # 中性话语，实际负面
        ("neutral", "还行", "disgust"): 0.6,
        ("neutral", "一般", "contempt"): 0.5,
        ("neutral", "就那样", "contempt"): 0.6,
        ("neutral", "还好", "boredom"): 0.4,
        ("neutral", "凑合", "disgust"): 0.5,
        ("neutral", "过得去", "boredom"): 0.4,

        # 夸张正面，实际负面
        ("exaggerated", "太厉害了", "contempt"): 0.8,
        ("exaggerated", "哇塞", "fear"): 0.5,
        ("exaggerated", "我的天", "fear"): 0.4,
        ("exaggerated", "天哪", "fear"): 0.3,
        ("exaggerated", "牛啊", "contempt"): 0.6,
    }

    # 反讽关键词
    IRONY_KEYWORDS = [
        "可真", "真是", "好是", "倒是", "还挺", "挺会",
        "会说话", "真会说", "可真行", "你可真是",
        "真行", "可真会", "可真会装", "装什么",
        "有什么好", "有什么好得意", "就你厉害",
        "你最棒", "你最厉害", "你可真行",
    ]

    # 矛盾情感词组
    CONTRADICTORY_PATTERNS = [
        ("开心", "哭"), ("高兴", "难受"), ("喜欢", "讨厌"),
        ("期待", "害怕"), ("相信", "怀疑"), ("爱", "恨"),
    ]

    # 反讽语气词
    IRONY_PARTICLES = ["啊", "呀", "呢", "哈", "呵", "嘞"]

    def detect(self, text: str, surface_emotion: str, intensity: float) -> IronyResult:
        """
        检测反讽

        Args:
            text: 输入文本
            surface_emotion: 表面情感
            intensity: 表面强度

        Returns:
            IronyResult: 反讽检测结果
        """
        clues = []
        confidence = 0.0
        true_emotion = None

        # 1. 检查反讽模式
        pattern_result = self._check_irony_patterns(text, surface_emotion)
        if pattern_result:
            clues.extend(pattern_result["clues"])
            confidence += pattern_result["confidence"] * 0.5

        # 2. 检查矛盾情感
        contradiction = self._check_contradictions(text)
        if contradiction:
            clues.append(contradiction)
            confidence += 0.3

        # 3. 检查语气词组合
        particle_result = self._check_irony_particles(text, surface_emotion)
        if particle_result:
            clues.extend(particle_result["clues"])
            confidence += particle_result["confidence"] * 0.3

        # 4. 检查标点异常
        punctuation_result = self._check_punctuation(text, surface_emotion)
        if punctuation_result:
            clues.extend(punctuation_result["clues"])
            confidence += punctuation_result["confidence"] * 0.2

        # 5. 检查情境矛盾
        context_result = self._check_context_contradiction(text, surface_emotion)
        if context_result:
            clues.append(context_result)
            confidence += 0.2

        # 判断是否反讽
        is_irony = confidence >= 0.4 and len(clues) >= 2

        if is_irony:
            # 根据表面情感推断真实情感
            true_emotion = self._infer_true_emotion(surface_emotion, clues)

        return IronyResult(
            is_irony=is_irony,
            surface_emotion=surface_emotion,
            true_emotion=true_emotion,
            confidence=min(1.0, confidence),
            clues=clues,
        )

    def _check_irony_patterns(self, text: str, surface_emotion: str) -> Optional[Dict]:
        """检查反讽模式"""
        clues = []
        confidence = 0.0

        # 检查反讽关键词
        for keyword in self.IRONY_KEYWORDS:
            if keyword in text:
                clues.append(f"包含反讽词: '{keyword}'")
                confidence += 0.2

        # 检查特定词组
        irony_phrases = {
            "可真": 0.4,
            "真是": 0.3,
            "好是": 0.4,
            "你可真是": 0.6,
            "会说话": 0.5,
            "可真行": 0.6,
        }

        for phrase, conf in irony_phrases.items():
            if phrase in text:
                clues.append(f"反讽词组: '{phrase}'")
                confidence += conf

        # 检查"挺好的"类
        if "挺" in text and ("好" in text or "行" in text):
            if surface_emotion in ["joy", "trust"]:
                clues.append("正面词+'挺'可能为反讽")
                confidence += 0.3

        if clues:
            return {"clues": clues, "confidence": min(1.0, confidence)}
        return None

    def _check_contradictions(self, text: str) -> Optional[str]:
        """检查矛盾情感"""
        for pos, neg in self.CONTRADICTORY_PATTERNS:
            if pos in text and neg in text:
                return f"矛盾情感词: '{pos}' 和 '{neg}' 同时出现"
        return None

    def _check_irony_particles(self, text: str, surface_emotion: str) -> Optional[Dict]:
        """检查反讽语气词组合"""
        clues = []
        confidence = 0.0

        particle_count = sum(1 for p in self.IRONY_PARTICLES if p in text)

        # 正面情感但有多个语气词
        if surface_emotion in ["joy", "trust"] and particle_count >= 2:
            clues.append(f"正面情感+{particle_count}个语气词")
            confidence += 0.3

        # "好是...啊" 结构
        if "好是" in text and "啊" in text:
            clues.append("'好是...啊'反讽结构")
            confidence += 0.4

        # "可真...啊" 结构
        if "可真" in text and "啊" in text:
            clues.append("'可真...啊'反讽结构")
            confidence += 0.4

        if clues:
            return {"clues": clues, "confidence": min(1.0, confidence)}
        return None

    def _check_punctuation(self, text: str, surface_emotion: str) -> Optional[Dict]:
        """检查标点异常"""
        clues = []
        confidence = 0.0

        # 问号+感叹号组合
        if "？" in text and "！" in text:
            if surface_emotion in ["joy", "trust"]:
                clues.append("问号+感叹号组合可能表示反讽")
                confidence += 0.2

        # 省略号在正面情感后
        if "..." in text or "。。" in text or "……" in text:
            if surface_emotion == "joy":
                clues.append("正面情感+省略号可能为反讽")
                confidence += 0.2

        if clues:
            return {"clues": clues, "confidence": min(1.0, confidence)}
        return None

    def _check_context_contradiction(self, text: str, surface_emotion: str) -> Optional[str]:
        """检查情境矛盾"""
        # "太开心了...才怪" 结构
        if "才怪" in text or "才有鬼" in text:
            return "否定结构'才怪'表示反讽"

        # "说什么...也不..." 结构
        if "说什么" in text and "也不" in text:
            return "'说什么...也不'结构表示反讽"

        return None

    def _infer_true_emotion(self, surface_emotion: str, clues: list) -> str:
        """根据线索推断真实情感"""
        # 如果表面是joy，可能是contempt或disgust
        if surface_emotion == "joy":
            if any("厉害" in c or "优秀" in c or "棒" in c for c in clues):
                return "contempt"
            return "disgust"

        # 如果表面是trust，可能是envy或anger
        if surface_emotion == "trust":
            return "anger"

        return "sadness"
