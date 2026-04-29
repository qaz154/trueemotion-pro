# -*- coding: utf-8 -*-
"""
Irony Detector Module
====================

反讽检测模块：检测文本中的反讽表达

反讽类型：
1. 克制型反讽 - "还行吧"（实际很满意）
2. 愤怒型反讽 - "真是太感谢了"（实际是愤怒）
3. 悲伤型反讽 - "太好了，又迟到了"（实际是无奈）
4. 恐惧型反讽 - "好怕怕哦"（实际是不屑）
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trueemotion.emotion.plutchik24 import EMOTION_DEFINITIONS


@dataclass
class IronyResult:
    """反讽检测结果"""
    is_irony: bool
    irony_type: str  # "restraint", "anger", "sadness", "fear", "unknown"
    surface_emotion: str
    true_emotion: str
    confidence: float
    markers: List[str]  # 检测到的反讽标记


class IronyDetector:
    """
    中文反讽检测器

    检测策略：
    1. 情感对比：文字正面 + 上下文负面 = 可能反讽
    2. 夸张标记："真是"、"太...了"等
    3. 上下文不一致：前后情感矛盾
    4. 语义反转：特定反讽词汇模式
    """

    # 反讽语义反转模式
    IRONY_PATTERNS = [
        # 克制型反讽
        (r"还行吧", "positive_restraint", "disgust", ["还行吧", "一般般", "就那样"]),
        (r"不错哦", "positive_restraint", "disgust", ["不错", "挺好", "可以"]),
        (r"挺好的", "positive_restraint", "disgust", ["挺好", "蛮好", "还好"]),
        (r"好吧", "reluctant", "sadness", ["好吧", "行吧", "算了吧"]),
        (r"随你", "indifferent", "sadness", ["随你", "随便", "无所谓"]),

        # 愤怒型反讽
        (r"真是太感谢了", "angry_sarcasm", "anger", ["太感谢了", "谢谢啊", "感谢"]),
        (r"你真行", "angry_sarcasm", "anger", ["你真行", "你厉害", "你可真棒"]),
        (r"跪谢", "angry_sarcasm", "anger", ["跪谢", "谢恩", "感恩"]),
        (r"真棒", "angry_sarcasm", "anger", ["真棒", "真厉害", "真好"]),
        (r"可真行", "angry_sarcasm", "anger", ["可真行", "可真行啊"]),

        # 悲伤型反讽
        (r"太好了", "sad_sarcasm", "sadness", ["太好了", "真好啊", "太美了"]),
        (r"没事没事", "sad_sarcasm", "sadness", ["没事", "没关系", "不要紧"]),
        (r"习惯了", "sad_sarcasm", "sadness", ["习惯了", "都这样", "算了"]),
        (r"反正", "resigned", "sadness", ["反正", "早晚", "终究"]),

        # 恐惧型反讽
        (r"好怕怕", "fear_sarcasm", "disgust", ["好怕怕", "吓死", "怕怕"]),
        (r"吓死我了", "fear_sarcasm", "disgust", ["吓死我了", "好吓人"]),

        # 讽刺型
        (r"呵呵", "cynical", "disgust", ["呵呵", "哈哈", "嘿嘿"]),
        (r"笑死", "cynical", "disgust", ["笑死", "笑死我了", "可笑"]),
        (r"可真是", "cynical", "disgust", ["可真是", "可真是啊"]),
        (r"有道理", "cynical", "disgust", ["有道理", "说得对"]),

        # 否定型反讽
        (r"不是.*吗", "sarcastic_question", "anger", ["不是吗", "难道不是", "不是么"]),
    ]

    # 夸张正面词（反讽标记）
    EXAGGERATED_POSITIVE = [
        "太棒了", "太厉害了", "太牛了", "太完美了",
        "简直了", "没谁了", "无敌了", "绝了"
    ]

    # 克制表达词
    RESTRAINT_MARKERS = [
        "还行", "一般", "就那样", "普通", "正常",
        "凑合", "还行吧", "一般般", "就那样吧"
    ]

    # 无奈表达词
    RESIGNATION_MARKERS = [
        "算了", "随便", "无所谓", "反正", "习惯了",
        "没所谓", "管他呢", "爱怎样怎样"
    ]

    # 表面正面但实际负面的词
    POSITIVE_SURFACE_NEGATIVE_TRUE = {
        "太好了": "sadness",
        "真棒": "anger",
        "不错": "disgust",
        "厉害": "anger",
        "佩服": "contempt",
        "谢了": "anger",
    }

    def __init__(self):
        self.patterns = []
        self._compile_patterns()

    def _compile_patterns(self):
        """编译正则表达式模式"""
        for pattern, irony_type, true_emotion, examples in self.IRONY_PATTERNS:
            self.patterns.append({
                "pattern": re.compile(pattern),
                "irony_type": irony_type,
                "true_emotion": true_emotion,
                "examples": examples
            })

    def detect(self, text: str, context_emotions: Optional[List[str]] = None) -> IronyResult:
        """
        检测文本是否包含反讽

        Args:
            text: 输入文本
            context_emotions: 上下文情感列表（前面几句话的情感）

        Returns:
            IronyResult: 反讽检测结果
        """
        text = text.strip()
        markers_found = []

        # 1. 模式匹配检测
        for p in self.patterns:
            if p["pattern"].search(text):
                markers_found.append(p["examples"][0] if p["examples"] else p["pattern"].pattern)
                confidence = self._calculate_pattern_confidence(text, p)
                surface_emotion = self._infer_surface_emotion(text, p)

                return IronyResult(
                    is_irony=True,
                    irony_type=p["irony_type"],
                    surface_emotion=surface_emotion,
                    true_emotion=p["true_emotion"],
                    confidence=confidence,
                    markers=markers_found
                )

        # 2. 夸张正面词检测
        for word in self.EXAGGERATED_POSITIVE:
            if word in text:
                # 如果包含感叹号加强，可能是真正面
                if "!" in text or "！" in text:
                    continue
                # 如果上下文是负面情感，可能是反讽
                if context_emotions and any(e in ["sadness", "anger", "fear"] for e in context_emotions[-2:]):
                    return IronyResult(
                        is_irony=True,
                        irony_type="contextual_sarcasm",
                        surface_emotion="joy",
                        true_emotion="sadness",
                        confidence=0.7,
                        markers=[word]
                    )

        # 3. 克制表达检测
        for marker in self.RESTRAINT_MARKERS:
            if marker in text:
                # 后面跟着正面词但语气低沉
                if "。" in text or "，" in text:
                    return IronyResult(
                        is_irony=True,
                        irony_type="restraint",
                        surface_emotion="neutral",
                        true_emotion=self._infer_true_emotion_from_context(text),
                        confidence=0.65,
                        markers=[marker]
                    )

        # 4. 无奈表达检测
        for marker in self.RESIGNATION_MARKERS:
            if marker in text:
                return IronyResult(
                    is_irony=True,
                    irony_type="resignation",
                    surface_emotion="neutral",
                    true_emotion="sadness",
                    confidence=0.75,
                    markers=[marker]
                )

        # 5. 上下文不一致检测
        if context_emotions and len(context_emotions) >= 2:
            # 连续正面情感后突然负面表达
            recent_positive = context_emotions[-3:]
            if all(e in ["joy", "anticipation", "trust"] for e in recent_positive):
                if any(word in text for word in ["算了", "随便", "无所谓"]):
                    return IronyResult(
                        is_irony=True,
                        irony_type="masked_frustration",
                        surface_emotion="neutral",
                        true_emotion="sadness",
                        confidence=0.6,
                        markers=["context_mismatch"]
                    )

        return IronyResult(
            is_irony=False,
            irony_type="none",
            surface_emotion=self._infer_surface_emotion(text),
            true_emotion=self._infer_surface_emotion(text),
            confidence=0.0,
            markers=[]
        )

    def _calculate_pattern_confidence(self, text: str, pattern_info: dict) -> float:
        """计算反讽检测置信度"""
        base_confidence = 0.7

        # 感叹号降低置信度（可能是真正面）
        if "!" in text or "！" in text:
            base_confidence -= 0.15

        # 问号增加置信度（反问句）
        if "?" in text or "？" in text:
            base_confidence += 0.1

        # 句号增加置信度（陈述句）
        if "。" in text:
            base_confidence += 0.1

        # 包含多个反讽词
        marker_count = sum(1 for m in pattern_info["examples"] if m in text)
        if marker_count > 1:
            base_confidence += 0.1

        return min(0.95, max(0.5, base_confidence))

    def _infer_surface_emotion(self, text: str, pattern_info: Optional[dict] = None) -> str:
        """推断表面情感"""
        if pattern_info:
            irony_type = pattern_info["irony_type"]
            if "positive" in irony_type or "restraint" in irony_type:
                return "joy"
            elif "angry" in irony_type:
                return "joy"
            elif "sad" in irony_type:
                return "joy"
            elif "fear" in irony_type:
                return "fear"

        # 基于文本特征推断
        if any(word in text for word in ["开心", "高兴", "棒", "好"]):
            return "joy"
        elif any(word in text for word in ["谢", "感谢", "感恩"]):
            return "joy"
        return "neutral"

    def _infer_true_emotion_from_context(self, text: str) -> str:
        """基于文本特征推断真实情感"""
        if any(word in text for word in ["算了", "随便", "无所谓", "反正"]):
            return "sadness"
        elif any(word in text for word in ["不错", "还行", "一般"]):
            return "disgust"
        return "sadness"

    def batch_detect(self, texts: List[str], context_emotions: Optional[List[List[str]]] = None) -> List[IronyResult]:
        """
        批量检测反讽

        Args:
            texts: 文本列表
            context_emotions: 每个文本的上下文情感列表

        Returns:
            List[IronyResult]: 反讽检测结果列表
        """
        results = []
        for i, text in enumerate(texts):
            ctx = context_emotions[i] if context_emotions and i < len(context_emotions) else None
            results.append(self.detect(text, ctx))
        return results


class IronyDatasetGenerator:
    """
    反讽数据集生成器

    生成真实的反讽训练样本
    """

    # 反讽场景模板
    IRONY_SCENARIOS = {
        "restraint_positive": {
            "description": "克制型正面反讽",
            "templates": [
                {"surface": "还行吧，就那样。", "true": "disgust", "context": "job_performance"},
                {"surface": "不错哦，一般般啦。", "true": "disgust", "context": "gift_received"},
                {"surface": "挺好的，正常发挥。", "true": "disgust", "context": "exam_result"},
            ]
        },
        "angry_sarcasm": {
            "description": "愤怒型反讽",
            "templates": [
                {"surface": "真是太感谢了，让我等了三个小时。", "true": "anger", "context": "customer_service"},
                {"surface": "你可真行，又把我的方案否决了。", "true": "anger", "context": "work_meeting"},
                {"surface": "行行行，你说的都对。", "true": "anger", "context": "argument"},
            ]
        },
        "sad_sarcasm": {
            "description": "悲伤型反讽",
            "templates": [
                {"surface": "太好了，又迟到了。", "true": "sadness", "context": "traffic"},
                {"surface": "没事没事，习惯了。", "true": "sadness", "context": "chronic_issue"},
                {"surface": "反正也没人会在意。", "true": "sadness", "context": "personal_issue"},
            ]
        },
        "fear_sarcasm": {
            "description": "不屑型反讽",
            "templates": [
                {"surface": "好怕怕哦，吓死人了呢。", "true": "disgust", "context": "mocking_fear"},
                {"surface": "哇，好厉害啊，真吓人。", "true": "disgust", "context": "mocking_strength"},
            ]
        },
        "cynical": {
            "description": "讽刺型",
            "templates": [
                {"surface": "呵呵，果然不出所料。", "true": "disgust", "context": "failed_promise"},
                {"surface": "笑死，这也行？", "true": "disgust", "context": "ridiculous_situation"},
                {"surface": "可真是人才啊。", "true": "contempt", "context": "ironic_praise"},
            ]
        }
    }

    def generate_samples(self, count_per_type: int = 50) -> List[Dict]:
        """
        生成反讽训练样本

        Returns:
            List[Dict]: 反讽样本列表
        """
        samples = []

        for irony_type, scenario in self.IRONY_SCENARIOS.items():
            templates = scenario["templates"]

            for _ in range(count_per_type):
                template = templates[_ % len(templates)]

                sample = {
                    "text": template["surface"],
                    "is_irony": True,
                    "irony_type": irony_type,
                    "surface_emotion": "joy",  # 反讽表面是正面情感
                    "true_emotion": template["true"],
                    "context": template["context"],
                    "scenario": scenario["description"]
                }
                samples.append(sample)

        return samples

    def get_test_samples(self) -> List[Dict]:
        """获取反讽测试样本"""
        test_samples = [
            # 克制型
            {"text": "还行吧，一般这种情况。", "expected": True, "true_emotion": "disgust"},
            {"text": "嗯，不错不错。", "expected": True, "true_emotion": "disgust"},
            {"text": "挺好的，正常水平。", "expected": True, "true_emotion": "disgust"},

            # 愤怒型
            {"text": "太谢谢了啊，等了俩小时。", "expected": True, "true_emotion": "anger"},
            {"text": "你可真是好样的。", "expected": True, "true_emotion": "anger"},
            {"text": "行行行，你说了算。", "expected": True, "true_emotion": "anger"},

            # 悲伤型
            {"text": "太好了，又加班到十二点。", "expected": True, "true_emotion": "sadness"},
            {"text": "没事，习惯了。", "expected": True, "true_emotion": "sadness"},
            {"text": "反正都这样了。", "expected": True, "true_emotion": "sadness"},

            # 讽刺型
            {"text": "呵呵，果然是你。", "expected": True, "true_emotion": "disgust"},
            {"text": "笑死，这剧本。", "expected": True, "true_emotion": "disgust"},

            # 真正面（非反讽）
            {"text": "太开心了！终于成功了！", "expected": False, "true_emotion": "joy"},
            {"text": "谢谢你的礼物，我很喜欢！", "expected": False, "true_emotion": "joy"},
            {"text": "真的假的？太棒了吧！", "expected": False, "true_emotion": "joy"},
        ]

        return test_samples


if __name__ == "__main__":
    # 测试反讽检测器
    detector = IronyDetector()

    test_texts = [
        "还行吧，就那样。",
        "真是太感谢了，让我等了这么久。",
        "太好了，又迟到了。",
        "好怕怕哦，吓死人了呢。",
        "呵呵，果然不出所料。",
        "太开心了！终于成功了！",  # 真正面
        "谢谢你的礼物！",  # 真正面
    ]

    print("=" * 60)
    print("反讽检测测试")
    print("=" * 60)

    for text in test_texts:
        result = detector.detect(text)
        print(f"\n文本: {text}")
        print(f"  反讽: {result.is_irony}")
        print(f"  类型: {result.irony_type}")
        print(f"  表面: {result.surface_emotion} -> 真实: {result.true_emotion}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  标记: {result.markers}")

    # 测试数据集生成器
    print("\n" + "=" * 60)
    print("反讽数据生成测试")
    print("=" * 60)

    generator = IronyDatasetGenerator()
    samples = generator.generate_samples(count_per_type=3)
    print(f"\n生成了 {len(samples)} 个反讽样本")

    test_samples = generator.get_test_samples()
    print(f"测试样本数: {len(test_samples)}")
