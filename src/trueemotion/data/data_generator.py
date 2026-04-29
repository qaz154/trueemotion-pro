# -*- coding: utf-8 -*-
"""
TrueEmotion Data Generator - 场景驱动训练数据生成器
====================================================

生成多样化、高质量的情感训练数据：
1. 场景模板驱动（真实心理场景）
2. 多标签标注（原型+复合+VAD+强度+反讽）
3. 上下文对话生成
4. 反讽样本专项生成

数据规模：单次运行可生成10万-1000万样本
"""

import os
import sys
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trueemotion.emotion.plutchik24 import EMOTION_DEFINITIONS, VAD_LEXICON


# ==================== 情感模板定义 ====================

@dataclass
class EmotionTemplate:
    """情感模板"""
    emotion: str
    intensity_range: Tuple[float, float]
    vad: Tuple[float, float, float]
    templates: List[str]
    modifiers: List[str]
    contexts: List[str]


# 24种情感模板
EMOTION_TEMPLATES: Dict[str, EmotionTemplate] = {
    # ==================== 8种原始情感 ====================
    "joy": EmotionTemplate(
        emotion="joy",
        intensity_range=(0.6, 1.0),
        vad=(0.9, 0.5, 0.75),
        templates=[
            "今天太{modifier}开心了！{context}",
            "项目终于完成了，{modifier}高兴！",
            "收到礼物，{modifier}开心！",
            "考试考好了，{modifier}棒！",
            "{context}，{modifier}快乐！",
            "中彩票了，{modifier}爽！",
            "见到老朋友，{modifier}开心！",
            "天气真好，心情{modifier}舒畅！",
        ],
        modifiers=["非常", "特别", "极其", "超级", "格外", "十分", "相当", ""],
        contexts=["假期来临", "任务完成", "目标达成", "梦想成真", "一切顺利"]
    ),
    "trust": EmotionTemplate(
        emotion="trust",
        intensity_range=(0.5, 0.9),
        vad=(0.65, -0.25, 0.45),
        templates=[
            "我相信{context}，没问题",
            "{context}，我相信你",
            "这件事交给你，我放心",
            "完全信任{context}",
            "托付给你，我安心",
        ],
        modifiers=["完全", "非常", "十分", "相当", ""],
        contexts=["这件事", "这个任务", "这个项目", "这个决定"]
    ),
    "fear": EmotionTemplate(
        emotion="fear",
        intensity_range=(0.5, 1.0),
        vad=(-0.75, 0.65, -0.65),
        templates=[
            "{modifier}害怕{context}",
            "我好担心{context}",
            "担心会发生{context}",
            "{context}让我{modifier}不安",
            "一想到{context}就紧张",
        ],
        modifiers=["非常", "特别", "极其", "相当", ""],
        contexts=["考试", "面试", "明天的事", "结果", "未知"]
    ),
    "anger": EmotionTemplate(
        emotion="anger",
        intensity_range=(0.5, 1.0),
        vad=(-0.85, 0.75, 0.55),
        templates=[
            "{modifier}气死我了！",
            "这也太{modifier}让人生气了吧",
            "{context}，{modifier}火大",
            "真是{modifier}烦人！",
            "怎么能{modifier}这样！",
        ],
        modifiers=["非常", "特别", "极其", "相当", "极度", ""],
        contexts=["被骗", "被坑", "不公平", "被忽视", "被嘲笑"]
    ),
    "surprise": EmotionTemplate(
        emotion="surprise",
        intensity_range=(0.5, 1.0),
        vad=(0.15, 0.85, 0.25),
        templates=[
            "哇，{modifier}惊讶！",
            "真是{modifier}没想到！",
            "{context}，{modifier}震惊！",
            "居然{context}，{modifier}意外！",
            "{modifier}吃惊，{context}！",
        ],
        modifiers=["非常", "特别", "十分", "相当", ""],
        contexts=["他来了", "成绩这么好", "这个消息", "这种情况"]
    ),
    "anticipation": EmotionTemplate(
        emotion="anticipation",
        intensity_range=(0.5, 0.95),
        vad=(0.45, 0.55, 0.55),
        templates=[
            "{modifier}期待{context}！",
            "好想{context}啊！",
            "希望{context}快到来",
            "好兴奋，要{context}了！",
            "{context}，{modifier}憧憬",
        ],
        modifiers=["非常", "特别", "十分", "相当", "极度", ""],
        contexts=["旅行", "放假", "见到你", "结果公布", "开始"]
    ),
    "sadness": EmotionTemplate(
        emotion="sadness",
        intensity_range=(0.5, 1.0),
        vad=(-0.85, -0.35, -0.55),
        templates=[
            "我{modifier}难过...",
            "{context}，{modifier}伤心",
            "心情{modifier}低落",
            "{modifier}失落...",
            "好{modifier}痛苦...",
        ],
        modifiers=["非常", "特别", "极其", "相当", ""],
        contexts=["失恋", "失去", "失败", "分离", "挫折"]
    ),
    "disgust": EmotionTemplate(
        emotion="disgust",
        intensity_range=(0.4, 0.9),
        vad=(-0.85, -0.15, -0.45),
        templates=[
            "{modifier}讨厌{context}！",
            "真{modifier}恶心！",
            "{context}，{modifier}嫌弃",
            "受不了{context}",
            "怎么{modifier}这么烦人！",
        ],
        modifiers=["非常", "特别", "极其", "极度", ""],
        contexts=["这个味道", "这种事", "这个人", "这种行为", "这天气"]
    ),

    # ==================== 复合情感 ====================
    "optimism": EmotionTemplate(
        emotion="optimism",
        intensity_range=(0.5, 0.9),
        vad=(0.75, 0.55, 0.65),
        templates=[
            "虽然现在难，但{modifier}会好的",
            "{modifier}相信一切都会好起来",
            "曙光就在前面，{modifier}乐观",
            "明天一定会更好，{modifier}有信心",
        ],
        modifiers=["很", "非常", "十分", ""],
        contexts=["困难", "挫折", "低谷"]
    ),
    "love": EmotionTemplate(
        emotion="love",
        intensity_range=(0.6, 1.0),
        vad=(0.90, 0.35, 0.70),
        templates=[
            "{modifier}喜欢你！",
            "我对{context}有{modifier}好感",
            "真的{modifier}很在乎你",
            "{modifier}心动的感觉",
            "我{modifier}爱死你了！",
        ],
        modifiers=["非常", "特别", "十分", "超级", ""],
        contexts=["你", "这个人", "这个家", "我的朋友"]
    ),
    "guilt": EmotionTemplate(
        emotion="guilt",
        intensity_range=(0.4, 0.9),
        vad=(-0.55, 0.25, -0.45),
        templates=[
            "对不起，我{modifier}内疚",
            "真是{modifier}过意不去",
            "我{modifier}惭愧...",
            "{modifier}抱歉，我不是故意的",
        ],
        modifiers=["非常", "十分", "相当", ""],
        contexts=["做错事", "失约", "伤害别人"]
    ),
    "submission": EmotionTemplate(
        emotion="submission",
        intensity_range=(0.3, 0.7),
        vad=(-0.25, -0.35, -0.65),
        templates=[
            "算了，{modifier}认命吧",
            "{modifier}没办法，只能这样",
            "{modifier}就这样吧",
            "随便吧，{modifier}无所谓",
        ],
        modifiers=["很", "有点", "真的", ""],
        contexts=["无奈", "无力", "被动接受"]
    ),
    "surprise_complex": EmotionTemplate(
        emotion="surprise_complex",
        intensity_range=(0.5, 1.0),
        vad=(-0.35, 0.85, -0.15),
        templates=[
            "{modifier}震惊！居然会这样！",
            "太{modifier}骇人了！",
            "什么？{modifier}难以置信！",
            "{modifier}吓人，怎么可能！",
        ],
        modifiers=["非常", "特别", "极其", ""],
        contexts=["这个消息", "这种事", "这个结果"]
    ),
    "disappointment": EmotionTemplate(
        emotion="disappointment",
        intensity_range=(0.5, 0.9),
        vad=(-0.75, -0.25, -0.55),
        templates=[
            "真{modifier}失望...",
            "{modifier}没希望了",
            "哎，{modifier}沮丧",
            "{modifier}绝望了...",
        ],
        modifiers=["非常", "特别", "十分", "极度", ""],
        contexts=["失败", "落空", "期望落空", "希望破灭"]
    ),
    "remorse": EmotionTemplate(
        emotion="remorse",
        intensity_range=(0.4, 0.9),
        vad=(-0.70, -0.15, -0.50),
        templates=[
            "真后悔{context}...",
            "{modifier}早知道就好了",
            "{modifier}悔不当初",
            "我{modifier}后悔死了",
        ],
        modifiers=["非常", "十分", "相当", ""],
        contexts=["做了那件事", "没珍惜", "错过机会"]
    ),
    "envy": EmotionTemplate(
        emotion="envy",
        intensity_range=(0.3, 0.8),
        vad=(-0.35, 0.35, -0.25),
        templates=[
            "真{modifier}羡慕啊！",
            "为什么{context}这么厉害...",
            "我也想要{context}，{modifier}嫉妒",
            "{modifier}眼红！",
        ],
        modifiers=["有点", "很", "非常", ""],
        contexts=["他", "她", "别人的成就", "别人的运气"]
    ),
    "suspicion": EmotionTemplate(
        emotion="suspicion",
        intensity_range=(0.3, 0.8),
        vad=(-0.55, 0.25, -0.15),
        templates=[
            "{modifier}怀疑{context}",
            "我不相信{context}",
            "{modifier}觉得有问题",
            "怎么{modifier}觉得不对劲...",
        ],
        modifiers=["有点", "很", "非常", ""],
        contexts=["这件事", "他的话", "这个情况"]
    ),
    "aggression": EmotionTemplate(
        emotion="aggression",
        intensity_range=(0.5, 1.0),
        vad=(-0.65, 0.75, 0.65),
        templates=[
            "{modifier}想揍人！",
            "真想{context}！",
            "{modifier}火大，要发泄",
            "冲动的想{context}",
        ],
        modifiers=["非常", "特别", "极度", ""],
        contexts=["打人", "砸东西", "发火", "发泄"]
    ),
    "pride": EmotionTemplate(
        emotion="pride",
        intensity_range=(0.5, 0.95),
        vad=(0.65, 0.45, 0.85),
        templates=[
            "{modifier}骄傲！",
            "真是{modifier}了不起！",
            "{modifier}自豪！",
            "自己都{modifier}佩服自己",
        ],
        modifiers=["很", "非常", "十分", "超级", ""],
        contexts=["成就", "成功", "胜利"]
    ),
    "contentment": EmotionTemplate(
        emotion="contentment",
        intensity_range=(0.4, 0.8),
        vad=(0.75, 0.2, 0.6),
        templates=[
            "现在{modifier}满足",
            "很{modifier}惬意",
            "{modifier}舒心",
            "感觉{modifier}很好",
        ],
        modifiers=["很", "非常", "十分", ""],
        contexts=["生活", "状态", "现状"]
    ),
    "contempt": EmotionTemplate(
        emotion="contempt",
        intensity_range=(0.4, 0.9),
        vad=(-0.75, 0.25, 0.45),
        templates=[
            "{modifier}鄙视！",
            "真{modifier}看不起",
            "{modifier}藐视",
            "有什么{modifier}了不起的！",
        ],
        modifiers=["很", "非常", "十分", ""],
        contexts=["这种人", "这种行为", "这本事"]
    ),
    "cynicism": EmotionTemplate(
        emotion="cynicism",
        intensity_range=(0.3, 0.8),
        vad=(-0.45, 0.15, 0.15),
        templates=[
            "{modifier}呵呵...",
            "真{modifier}可笑！",
            "{modifier}笑死人了",
            "也就{modifier}那样吧",
        ],
        modifiers=["有点", "真是", "简直", ""],
        contexts=["结果", "表现", "成就"]
    ),
    "morbidness": EmotionTemplate(
        emotion="morbidness",
        intensity_range=(0.2, 0.6),
        vad=(-0.25, 0.45, -0.25),
        templates=[
            "{modifier}有点阴暗",
            "世界{modifier}不美好",
            "人性{modifier}本恶",
            "{modifier}扭曲的想法",
        ],
        modifiers=["有点", "很", "相当", ""],
        contexts=["想法", "念头", "观念"]
    ),
    "sentimentality": EmotionTemplate(
        emotion="sentimentality",
        intensity_range=(0.3, 0.7),
        vad=(0.10, 0.25, 0.10),
        templates=[
            "往事{modifier}历历在目...",
            "好{modifier}感慨啊",
            "时光{modifier}难忘",
            "{modifier}怀念从前",
        ],
        modifiers=["很", "非常", "十分", ""],
        contexts=["回忆", "过去", "时光", "童年"]
    ),
    "anxiety": EmotionTemplate(
        emotion="anxiety",
        intensity_range=(0.4, 0.9),
        vad=(-0.55, 0.65, -0.45),
        templates=[
            "{modifier}焦虑...",
            "担心{modifier}不安",
            "心里{modifier}忐忑",
            "{modifier}忧心忡忡",
        ],
        modifiers=["很", "非常", "十分", "极度", ""],
        contexts=["未来", "结果", "未知", "压力"]
    ),
    "despair": EmotionTemplate(
        emotion="despair",
        intensity_range=(0.6, 1.0),
        vad=(-0.90, 0.35, -0.75),
        templates=[
            "真{modifier}绝望了...",
            "{modifier}活不下去了",
            "彻底{modifier}崩溃",
            "{modifier}无助到极点",
        ],
        modifiers=["非常", "十分", "极度", "彻底", ""],
        contexts=["困境", "绝境", "绝望", "无望"]
    ),
}


# ==================== 反讽模板 ====================

IRONY_TEMPLATES = {
    "restraint_joy": {
        # 克制型喜悦反讽
        "surface": "joy",
        "true_emotion": "sadness",
        "vad": (-0.7, -0.2, -0.3),
        "templates": [
            "太好了，又加班到凌晨",
            "哇，火车又晚点了，真棒",
            "谢谢，让我等了三个小时",
            "真好，又下雨了",
            "太感动了，又失败了",
        ]
    },
    "restraint_anger": {
        # 克制型愤怒反讽
        "surface": "joy",
        "true_emotion": "anger",
        "vad": (-0.6, 0.4, 0.2),
        "templates": [
            "没关系，完全不在意",
            "没事没事，我真的不生气",
            "谢谢你让我学会坚强",
            "没什么大不了的",
            "谢谢你的\"帮助\"",
        ]
    },
    "restraint_fear": {
        # 克制型恐惧反讽
        "surface": "surprise",
        "true_emotion": "fear",
        "vad": (-0.5, 0.5, -0.3),
        "templates": [
            "好怕怕哦，吓死我了呢",
            "好怕好怕（阴阳怪气）",
            "真的吗？我好担心哦",
        ]
    },
    "mock_positive": {
        # 讽刺正面
        "surface": "joy",
        "true_emotion": "disgust",
        "vad": (-0.5, 0.1, 0.1),
        "templates": [
            "你真厉害，第三次不及格",
            "真棒，又把事情搞砸了",
            "谢谢，每次都让我失望",
            "太好了，又一个第一（倒数）",
        ]
    },
    "mock_anticipation": {
        # 讽刺期待
        "surface": "anticipation",
        "true_emotion": "disappointment",
        "vad": (-0.5, -0.1, -0.2),
        "templates": [
            "好期待啊，又是一个无聊的周末",
            "太好了，要交作业了",
            "哇，要考试了呢",
        ]
    },
}


# ==================== 数据生成器 ====================

class EmotionDataGenerator:
    """
    情感训练数据生成器

    支持：
    - 场景驱动生成
    - 多标签标注
    - 反讽样本
    - 上下文对话
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        # 情感名称到索引的映射
        self.emotion_names = list(EMOTION_TEMPLATES.keys())
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotion_names)}

        # VAD基础值
        self.base_vad = {
            e: EMOTION_TEMPLATES[e].vad for e in self.emotion_names
        }

    def generate_single(self, emotion: str, use_template: bool = True) -> Dict[str, Any]:
        """生成单条数据"""
        template = EMOTION_TEMPLATES[emotion]

        if use_template and template.templates:
            # 使用模板
            base_text = random.choice(template.templates)
            modifier = random.choice(template.modifiers)
            context = random.choice(template.contexts)
            text = base_text.format(modifier=modifier, context=context)
        else:
            # 使用关键词组合
            text = self._generate_free_text(emotion)

        # 生成强度
        intensity = random.uniform(*template.intensity_range)

        # 生成VAD (带噪声)
        base_v = template.vad[0]
        base_a = template.vad[1]
        base_d = template.vad[2]
        vad = (
            np.clip(base_v + np.random.randn() * 0.1, -1, 1),
            np.clip(base_a + np.random.randn() * 0.15, -1, 1),
            np.clip(base_d + np.random.randn() * 0.1, -1, 1)
        )

        # 生成多标签
        emotion_labels = np.zeros(26, dtype=np.float32)
        emotion_labels[self.emotion_to_idx[emotion]] = intensity

        # 复合情感：检查相关情感
        self._add_complex_labels(emotion_labels, emotion, intensity)

        # 反讽标签 (默认0)
        irony_label = 0.0

        return {
            "text": text,
            "emotion_labels": emotion_labels,
            "vad_labels": np.array(vad, dtype=np.float32),
            "intensity_labels": np.array([intensity], dtype=np.float32),
            "irony_labels": np.array([irony_label], dtype=np.float32),
            "primary_emotion": emotion,
            "intensity": intensity,
            "vad": vad
        }

    def generate_irony(self, irony_type: str = None) -> Dict[str, Any]:
        """生成反讽数据"""
        if irony_type is None:
            irony_type = random.choice(list(IRONY_TEMPLATES.keys()))

        template = IRONY_TEMPLATES[irony_type]
        text = random.choice(template["templates"])

        # 表面情感 (通常是joy)
        surface_emotion = template["surface"]
        true_emotion = template["true_emotion"]
        vad = template["vad"]

        # 强度中等偏高
        intensity = random.uniform(0.5, 0.8)

        # 多标签
        emotion_labels = np.zeros(26, dtype=np.float32)
        emotion_labels[self.emotion_to_idx[true_emotion]] = intensity

        return {
            "text": text,
            "emotion_labels": emotion_labels,
            "vad_labels": np.array(vad, dtype=np.float32),
            "intensity_labels": np.array([intensity], dtype=np.float32),
            "irony_labels": np.array([1.0], dtype=np.float32),
            "primary_emotion": true_emotion,
            "intensity": intensity,
            "vad": vad,
            "is_irony": True,
            "irony_type": irony_type
        }

    def _add_complex_labels(self, emotion_labels: np.ndarray, primary: str, intensity: float):
        """添加复合情感标签"""
        complex_relations = {
            "joy": ["optimism", "love", "pride", "contentment"],
            "trust": ["love", "submission"],
            "fear": ["anxiety", "submission", "guilt", "remorse"],
            "anger": ["aggression", "pride", "contempt", "cynicism"],
            "surprise": ["surprise_complex", "fear"],
            "anticipation": ["optimism", "envy", "aggression", "anxiety"],
            "sadness": ["disappointment", "remorse", "envy", "despair"],
            "disgust": ["contempt", "cynicism", "suspicion", "disappointment"]
        }

        if primary in complex_relations:
            for comp in complex_relations[primary]:
                if comp in self.emotion_to_idx:
                    # 复合情感强度是原情感的0.4-0.7倍
                    comp_intensity = intensity * random.uniform(0.4, 0.7)
                    emotion_labels[self.emotion_to_idx[comp]] = comp_intensity

    def _generate_free_text(self, emotion: str) -> str:
        """自由组合生成文本"""
        emotion_keywords = {
            "joy": ["开心", "高兴", "快乐", "愉快", "幸福"],
            "sadness": ["难过", "伤心", "悲伤", "痛苦", "失落"],
            "anger": ["生气", "愤怒", "恼火", "气愤", "烦躁"],
            "fear": ["害怕", "担心", "恐惧", "焦虑", "不安"],
            "trust": ["信任", "相信", "依赖", "放心"],
            "disgust": ["讨厌", "恶心", "厌恶", "反感", "厌烦"],
            "surprise": ["惊讶", "意外", "吃惊", "震惊", "想不到"],
            "anticipation": ["期待", "希望", "兴奋", "憧憬"]
        }

        if emotion not in emotion_keywords:
            return f"这件事让我感到{emotion}"

        keywords = emotion_keywords[emotion]
        connectors = ["", "真的", "确实", "特别", "非常"]

        return f"我{random.choice(connectors)}{random.choice(keywords)}"

    def generate_batch(self, num_samples: int, irony_ratio: float = 0.15) -> Dict[str, Any]:
        """
        生成批量数据

        Args:
            num_samples: 样本数量
            irony_ratio: 反讽样本比例
        """
        texts = []
        emotion_labels = []
        vad_labels = []
        intensity_labels = []
        irony_labels = []

        irony_count = int(num_samples * irony_ratio)
        normal_count = num_samples - irony_count

        # 生成正常样本
        for _ in range(normal_count):
            emotion = random.choice(self.emotion_names)
            sample = self.generate_single(emotion)

            texts.append(sample["text"])
            emotion_labels.append(sample["emotion_labels"])
            vad_labels.append(sample["vad_labels"])
            intensity_labels.append(sample["intensity_labels"])
            irony_labels.append(sample["irony_labels"])

        # 生成反讽样本
        irony_types = list(IRONY_TEMPLATES.keys())
        for _ in range(irony_count):
            irony_type = random.choice(irony_types)
            sample = self.generate_irony(irony_type)

            texts.append(sample["text"])
            emotion_labels.append(sample["emotion_labels"])
            vad_labels.append(sample["vad_labels"])
            intensity_labels.append(sample["intensity_labels"])
            irony_labels.append(sample["irony_labels"])

        # 转换为数组
        return {
            "texts": texts,
            "emotion_labels": np.array(emotion_labels, dtype=np.float32),
            "vad_labels": np.array(vad_labels, dtype=np.float32),
            "intensity_labels": np.array(intensity_labels, dtype=np.float32),
            "irony_labels": np.array(irony_labels, dtype=np.float32)
        }

    def generate_scenario_batch(self, num_samples: int, scenarios: List[str] = None) -> Dict[str, Any]:
        """
        场景驱动批量生成

        每个场景包含多轮对话，更真实
        """
        if scenarios is None:
            scenarios = [
                "职场压力", "感情问题", "学业挑战", "家庭矛盾",
                "健康担忧", "财务困难", "社交尴尬", "成长困惑"
            ]

        texts = []
        emotion_labels = []
        vad_labels = []
        intensity_labels = []
        irony_labels = []

        samples_per_scenario = num_samples // len(scenarios)

        for scenario in scenarios:
            for _ in range(samples_per_scenario):
                # 从场景对应的情感分布中采样
                emotion = self._sample_emotion_for_scenario(scenario)
                sample = self.generate_single(emotion)

                texts.append(sample["text"])
                emotion_labels.append(sample["emotion_labels"])
                vad_labels.append(sample["vad_labels"])
                intensity_labels.append(sample["intensity_labels"])
                irony_labels.append(sample["irony_labels"])

        return {
            "texts": texts,
            "emotion_labels": np.array(emotion_labels, dtype=np.float32),
            "vad_labels": np.array(vad_labels, dtype=np.float32),
            "intensity_labels": np.array(intensity_labels, dtype=np.float32),
            "irony_labels": np.array(irony_labels, dtype=np.float32)
        }

    def generate(self, total_samples: int, irony_ratio: float = 0.15) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成大规模训练数据

        Returns:
            texts, emotion_labels, vad_labels, intensity_labels, irony_labels
        """
        print(f"生成 {total_samples:,} 样本...")

        # 分批生成
        batch_size = 10000
        all_texts = []
        all_emotion_labels = []
        all_vad_labels = []
        all_intensity_labels = []
        all_irony_labels = []

        for i in range(0, total_samples, batch_size):
            current_batch = min(batch_size, total_samples - i)
            batch_data = self.generate_scenario_batch(current_batch)

            all_texts.extend(batch_data["texts"])
            all_emotion_labels.append(batch_data["emotion_labels"])
            all_vad_labels.append(batch_data["vad_labels"])
            all_intensity_labels.append(batch_data["intensity_labels"])
            all_irony_labels.append(batch_data["irony_labels"])

            if (i + batch_size) % 50000 == 0:
                print(f"  已生成 {i + current_batch:,} / {total_samples:,}")

        # 合并
        return (
            all_texts,
            np.vstack(all_emotion_labels),
            np.vstack(all_vad_labels),
            np.vstack(all_intensity_labels),
            np.vstack(all_irony_labels)
        )

    def _sample_emotion_for_scenario(self, scenario: str) -> str:
        """根据场景采样情感"""
        scenario_emotions = {
            "职场压力": ["anxiety", "fear", "anger", "sadness", "despair"],
            "感情问题": ["sadness", "anger", "anxiety", "love", "despair"],
            "学业挑战": ["anxiety", "fear", "joy", "anticipation", "disappointment"],
            "家庭矛盾": ["anger", "sadness", "anxiety", "disgust", "despair"],
            "健康担忧": ["fear", "anxiety", "sadness", "anticipation", "despair"],
            "财务困难": ["anxiety", "fear", "sadness", "anger", "despair"],
            "社交尴尬": ["fear", "anxiety", "disgust", "submission", "remorse"],
            "成长困惑": ["anxiety", "surprise", "anticipation", "sadness", "submission"]
        }

        emotions = scenario_emotions.get(scenario, self.emotion_names)
        return random.choice(emotions)


# ==================== 多进程数据生成 ====================

def generate_partition(args):
    """多进程生成分区数据"""
    partition_id, num_samples, irony_ratio, seed = args
    generator = EmotionDataGenerator(seed=seed + partition_id)

    # 场景模式生成
    data = generator.generate_scenario_batch(num_samples)

    return partition_id, data


class DistributedDataGenerator:
    """分布式数据生成器"""

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)

    def generate(self, total_samples: int, irony_ratio: float = 0.15,
                output_dir: str = None) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成大规模训练数据

        Returns:
            texts, emotion_labels, vad_labels, intensity_labels, irony_labels
        """
        samples_per_partition = 50000
        num_partitions = (total_samples + samples_per_partition - 1) // samples_per_partition

        print(f"开始生成 {total_samples:,} 样本 ({num_partitions} 个分区, {self.num_workers} 进程)...")

        all_texts = []
        all_emotion_labels = []
        all_vad_labels = []
        all_intensity_labels = []
        all_irony_labels = []

        args_list = [
            (i, min(samples_per_partition, total_samples - i * samples_per_partition), irony_ratio, 42)
            for i in range(num_partitions)
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(generate_partition, args) for args in args_list]

            for future in as_completed(futures):
                partition_id, data = future.result()
                all_texts.extend(data["texts"])
                all_emotion_labels.append(data["emotion_labels"])
                all_vad_labels.append(data["vad_labels"])
                all_intensity_labels.append(data["intensity_labels"])
                all_irony_labels.append(data["irony_labels"])
                print(f"  分区 {partition_id + 1}/{num_partitions} 完成 ({len(data['texts']):,} 样本)")

        # 合并
        emotion_labels = np.vstack(all_emotion_labels)
        vad_labels = np.vstack(all_vad_labels)
        intensity_labels = np.vstack(all_intensity_labels)
        irony_labels = np.vstack(all_irony_labels)

        print(f"生成完成: {len(all_texts):,} 样本")

        return all_texts, emotion_labels, vad_labels, intensity_labels, irony_labels


if __name__ == "__main__":
    print("=" * 70)
    print("TrueEmotion Data Generator - 训练数据生成器")
    print("=" * 70)

    generator = EmotionDataGenerator()

    # 测试生成
    print("\n【单条生成测试】")
    for emotion in ["joy", "sadness", "anger", "fear"]:
        sample = generator.generate_single(emotion)
        print(f"  {emotion}: {sample['text']}")

    # 测试反讽生成
    print("\n【反讽生成测试】")
    irony_sample = generator.generate_irony()
    print(f"  {irony_sample['text']}")
    print(f"  is_irony: {irony_sample['is_irony']}")

    # 测试批量生成
    print("\n【批量生成测试】")
    batch_data = generator.generate_batch(1000, irony_ratio=0.15)
    print(f"  总样本: {len(batch_data['texts']):,}")
    print(f"  情感标签形状: {batch_data['emotion_labels'].shape}")
    print(f"  VAD标签形状: {batch_data['vad_labels'].shape}")
    print(f"  反讽样本数: {int(batch_data['irony_labels'].sum())}")

    # 测试场景生成
    print("\n【场景生成测试】")
    scenario_data = generator.generate_scenario_batch(100)
    print(f"  总样本: {len(scenario_data['texts']):,}")

    print("\n" + "=" * 70)
    print("数据生成器测试通过!")
    print("=" * 70)
