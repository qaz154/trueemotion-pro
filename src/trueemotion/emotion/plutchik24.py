# -*- coding: utf-8 -*-
"""
Plutchik 24 Emotion Ontology + VAD Dimensions
=============================================

TrueEmotion情感本体：24种情感（8原始+16复合）+ VAD连续维度

基于Robert Plutchik的情感轮盘理论
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class PrimaryEmotion(Enum):
    """8种原始情感"""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    ANTICIPATION = "anticipation"
    SADNESS = "sadness"
    DISGUST = "disgust"


class ComplexEmotion(Enum):
    """16种复合情感"""
    OPTIMISM = "optimism"           # 乐观: joy + anticipation
    LOVE = "love"                   # 爱: joy + trust
    GUILT = "guilt"                 # 内疚: joy + fear
    SUBMISSION = "submission"         # 顺从: trust + fear
    SURPRISE_COMPLEX = "surprise_complex"  # 意外: fear + surprise
    DISAPPOINTMENT = "disappointment"  # 失望: sadness + disgust
    REMORSE = "remorse"             # 后悔: sadness + fear
    ENVY = "envy"                   # 羡慕: sadness + anticipation
    SUSPICION = "suspicion"         # 怀疑: fear + disgust
    AGGRESSION = "aggression"        # 攻击: anger + anticipation
    PRIDE = "pride"                 # 骄傲: anger + joy
    CONTENTMENT = "contentment"     # 满足: joy + trust + anticipation
    CONTEMPT = "contempt"           # 蔑视: anger + disgust
    CYNICISM = "cynicism"           # 讽刺: disgust + anticipation
    MORBIDNESS = "morbidness"       # 病态: fear + joy
    SENTIMENTALITY = "sentimentality"  # 多愁善感: joy + sadness
    ANXIETY = "anxiety"             # 焦虑: fear + anticipation
    DESPAIR = "despair"             # 绝望: sadness + anger + fear


@dataclass(frozen=True)
class EmotionDefinition:
    """情感定义"""
    name: str
    chinese: str
    primary_components: Tuple[PrimaryEmotion, ...]
    vad: Tuple[float, float, float]  # Valence, Arousal, Dominance
    keywords: Tuple[str, ...]
    intensity_levels: Dict[str, str]  # {high, medium, low} -> expression


# 24种情感完整定义
EMOTION_DEFINITIONS: Dict[str, EmotionDefinition] = {
    # ==================== 8种原始情感 ====================
    "joy": EmotionDefinition(
        name="joy",
        chinese="喜",
        primary_components=(PrimaryEmotion.JOY,),
        vad=(0.95, 0.45, 0.75),  # 效价很高，唤醒度中等，支配度中高
        keywords=("开心", "高兴", "快乐", "喜悦", "愉快", "欢乐", "棒", "太棒了", "开心", "快乐", "幸福", "美好", "happy", "joy"),
        intensity_levels={
            "high": "狂喜",
            "medium": "开心",
            "low": "微微开心"
        }
    ),
    "trust": EmotionDefinition(
        name="trust",
        chinese="信",
        primary_components=(PrimaryEmotion.TRUST,),
        vad=(0.65, -0.25, 0.45),
        keywords=("信任", "相信", "依赖", "放心", "托付", "信赖"),
        intensity_levels={
            "high": "完全信任",
            "medium": "信任",
            "low": "有点信任"
        }
    ),
    "fear": EmotionDefinition(
        name="fear",
        chinese="惧",
        primary_components=(PrimaryEmotion.FEAR,),
        vad=(-0.75, 0.65, -0.65),
        keywords=("害怕", "担心", "恐惧", "焦虑", "不安", "紧张", "怕", "慌", "worried", "fear"),
        intensity_levels={
            "high": "恐惧",
            "medium": "害怕",
            "low": "担心"
        }
    ),
    "anger": EmotionDefinition(
        name="anger",
        chinese="怒",
        primary_components=(PrimaryEmotion.ANGER,),
        vad=(-0.85, 0.75, 0.55),
        keywords=("生气", "愤怒", "气愤", "恼火", "烦躁", "怒", "火", "气", "angry"),
        intensity_levels={
            "high": "暴怒",
            "medium": "生气",
            "low": "不满"
        }
    ),
    "surprise": EmotionDefinition(
        name="surprise",
        chinese="惊",
        primary_components=(PrimaryEmotion.SURPRISE,),
        vad=(0.15, 0.85, 0.25),
        keywords=("惊讶", "意外", "吃惊", "震惊", "想不到", "wow", "哇", "啥", "surprised"),
        intensity_levels={
            "high": "震惊",
            "medium": "惊讶",
            "low": "意外"
        }
    ),
    "anticipation": EmotionDefinition(
        name="anticipation",
        chinese="望",
        primary_components=(PrimaryEmotion.ANTICIPATION,),
        vad=(0.45, 0.55, 0.55),
        keywords=("期待", "盼望", "希望", "兴奋", "憧憬", "希望", "想", "anticipation", "hope"),
        intensity_levels={
            "high": "极度期待",
            "medium": "期待",
            "low": "有点期待"
        }
    ),
    "sadness": EmotionDefinition(
        name="sadness",
        chinese="哀",
        primary_components=(PrimaryEmotion.SADNESS,),
        vad=(-0.85, -0.35, -0.55),
        keywords=("难过", "伤心", "悲伤", "沮丧", "失落", "哀", "痛苦", "sad", "unhappy", "唉"),
        intensity_levels={
            "high": "悲痛",
            "medium": "难过",
            "low": "失落"
        }
    ),
    "disgust": EmotionDefinition(
        name="disgust",
        chinese="恶",
        primary_components=(PrimaryEmotion.DISGUST,),
        vad=(-0.85, -0.15, -0.45),
        keywords=("恶心", "讨厌", "嫌弃", "厌恶", "反感", "烦", "厌", "disgusted"),
        intensity_levels={
            "high": "厌恶",
            "medium": "讨厌",
            "low": "嫌弃"
        }
    ),

    # ==================== 16种复合情感 ====================
    "optimism": EmotionDefinition(
        name="optimism",
        chinese="乐",
        primary_components=(PrimaryEmotion.JOY, PrimaryEmotion.ANTICIPATION),
        vad=(0.75, 0.55, 0.65),
        keywords=("乐观", "有信心", "会好的", "曙光", "希望"),
        intensity_levels={
            "high": "非常乐观",
            "medium": "乐观",
            "low": "还行"
        }
    ),
    "love": EmotionDefinition(
        name="love",
        chinese="爱",
        primary_components=(PrimaryEmotion.JOY, PrimaryEmotion.TRUST),
        vad=(0.90, 0.35, 0.70),
        keywords=("爱", "喜欢", "心动", "在乎", "喜欢", "爱慕"),
        intensity_levels={
            "high": "深爱",
            "medium": "喜欢",
            "low": "好感"
        }
    ),
    "guilt": EmotionDefinition(
        name="guilt",
        chinese="愧",
        primary_components=(PrimaryEmotion.JOY, PrimaryEmotion.FEAR),
        vad=(-0.55, 0.25, -0.45),
        keywords=("内疚", "惭愧", "抱歉", "对不起", "过意不去"),
        intensity_levels={
            "high": "非常内疚",
            "medium": "内疚",
            "low": "有点愧疚"
        }
    ),
    "submission": EmotionDefinition(
        name="submission",
        chinese="从",
        primary_components=(PrimaryEmotion.TRUST, PrimaryEmotion.FEAR),
        vad=(-0.25, -0.35, -0.65),
        keywords=("顺从", "服从", "认命", "无奈", "算了"),
        intensity_levels={
            "high": "完全顺从",
            "medium": "顺从",
            "low": "勉强接受"
        }
    ),
    "surprise_complex": EmotionDefinition(
        name="surprise_complex",
        chinese="骇",
        primary_components=(PrimaryEmotion.FEAR, PrimaryEmotion.SURPRISE),
        vad=(-0.35, 0.85, -0.15),
        keywords=("震惊", "骇人", "惊人", "想不到", "吓人"),
        intensity_levels={
            "high": "惊恐",
            "medium": "震惊",
            "low": "惊讶"
        }
    ),
    "disappointment": EmotionDefinition(
        name="disappointment",
        chinese="失望",
        primary_components=(PrimaryEmotion.SADNESS, PrimaryEmotion.DISGUST),
        vad=(-0.75, -0.25, -0.55),
        keywords=("失望", "沮丧", "绝望", "没希望", "完了"),
        intensity_levels={
            "high": "极度失望",
            "medium": "失望",
            "low": "有点失望"
        }
    ),
    "remorse": EmotionDefinition(
        name="remorse",
        chinese="悔",
        primary_components=(PrimaryEmotion.SADNESS, PrimaryEmotion.FEAR),
        vad=(-0.70, -0.15, -0.50),
        keywords=("后悔", "悔恨", "懊悔", "早知道", "悔不当初"),
        intensity_levels={
            "high": "悔恨",
            "medium": "后悔",
            "low": "有点后悔"
        }
    ),
    "envy": EmotionDefinition(
        name="envy",
        chinese="羡",
        primary_components=(PrimaryEmotion.SADNESS, PrimaryEmotion.ANTICIPATION),
        vad=(-0.35, 0.35, -0.25),
        keywords=("羡慕", "嫉妒眼红", "眼热", "酸", "凭啥"),
        intensity_levels={
            "high": "非常嫉妒",
            "medium": "羡慕",
            "low": "有点眼红"
        }
    ),
    "suspicion": EmotionDefinition(
        name="suspicion",
        chinese="疑",
        primary_components=(PrimaryEmotion.FEAR, PrimaryEmotion.DISGUST),
        vad=(-0.55, 0.25, -0.15),
        keywords=("怀疑", "猜疑", "不信任", "质疑", "不信"),
        intensity_levels={
            "high": "非常怀疑",
            "medium": "怀疑",
            "low": "有点怀疑"
        }
    ),
    "aggression": EmotionDefinition(
        name="aggression",
        chinese="攻",
        primary_components=(PrimaryEmotion.ANGER, PrimaryEmotion.ANTICIPATION),
        vad=(-0.65, 0.75, 0.65),
        keywords=("攻击", "挑衅", "冲动", "想打人", "火大"),
        intensity_levels={
            "high": "暴怒",
            "medium": "愤怒",
            "low": "不满"
        }
    ),
    "pride": EmotionDefinition(
        name="pride",
        chinese="傲",
        primary_components=(PrimaryEmotion.ANGER, PrimaryEmotion.JOY),
        vad=(0.65, 0.45, 0.85),
        keywords=("骄傲", "自豪", "得意", "厉害", "了不起"),
        intensity_levels={
            "high": "骄傲自满",
            "medium": "自豪",
            "low": "有点得意"
        }
    ),
    "contempt": EmotionDefinition(
        name="contempt",
        chinese="蔑",
        primary_components=(PrimaryEmotion.ANGER, PrimaryEmotion.DISGUST),
        vad=(-0.75, 0.25, 0.45),
        keywords=("蔑视", "鄙视", "看不起", "不屑", "垃圾"),
        intensity_levels={
            "high": "极度蔑视",
            "medium": "鄙视",
            "low": "看不起"
        }
    ),
    "cynicism": EmotionDefinition(
        name="cynicism",
        chinese="讽",
        primary_components=(PrimaryEmotion.DISGUST, PrimaryEmotion.ANTICIPATION),
        vad=(-0.45, 0.15, 0.15),
        keywords=("讽刺", "冷嘲", "呵呵", "笑死", "真行"),
        intensity_levels={
            "high": "极度讽刺",
            "medium": "讽刺",
            "low": "冷嘲热讽"
        }
    ),
    "morbidness": EmotionDefinition(
        name="morbidness",
        chinese="扭曲",
        primary_components=(PrimaryEmotion.FEAR, PrimaryEmotion.JOY),
        vad=(-0.25, 0.45, -0.25),
        keywords=("病态", "扭曲", "阴暗", "变态"),
        intensity_levels={
            "high": "极度病态",
            "medium": "病态",
            "low": "有点阴暗"
        }
    ),
    "sentimentality": EmotionDefinition(
        name="sentimentality",
        chinese="愁",
        primary_components=(PrimaryEmotion.JOY, PrimaryEmotion.SADNESS),
        vad=(0.10, 0.25, 0.10),
        keywords=("多愁善感", "感慨", "怀旧", "往事", "时光"),
        intensity_levels={
            "high": "非常感慨",
            "medium": "感慨",
            "low": "有点怀念"
        }
    ),
    "anxiety": EmotionDefinition(
        name="anxiety",
        chinese="忧",
        primary_components=(PrimaryEmotion.FEAR, PrimaryEmotion.ANTICIPATION),
        vad=(-0.55, 0.65, -0.45),
        keywords=("焦虑", "忧心", "担忧", "忐忑", "不安"),
        intensity_levels={
            "high": "极度焦虑",
            "medium": "焦虑",
            "low": "担忧"
        }
    ),
    "despair": EmotionDefinition(
        name="despair",
        chinese="绝",
        primary_components=(PrimaryEmotion.SADNESS, PrimaryEmotion.ANGER, PrimaryEmotion.FEAR),
        vad=(-0.90, 0.35, -0.75),
        keywords=("绝望", "崩溃", "活不下去", "完了", "无助"),
        intensity_levels={
            "high": "彻底绝望",
            "medium": "绝望",
            "low": "很无助"
        }
    ),
}


# VAD情感词典：用于基于词典的情感分析
VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
    # 高Valence (+)
    "开心": (0.9, 0.7, 0.8),
    "高兴": (0.85, 0.65, 0.75),
    "快乐": (0.9, 0.6, 0.7),
    "喜悦": (0.85, 0.55, 0.7),
    "愉快": (0.8, 0.5, 0.65),
    "幸福": (0.95, 0.4, 0.75),
    "满足": (0.75, 0.2, 0.6),
    "兴奋": (0.8, 0.9, 0.6),
    "激动": (0.75, 0.85, 0.55),
    "棒": (0.85, 0.6, 0.5),
    "优秀": (0.8, 0.5, 0.7),

    # 低Valence (-)
    "难过": (-0.8, -0.3, -0.5),
    "伤心": (-0.85, -0.4, -0.55),
    "悲伤": (-0.85, -0.35, -0.5),
    "痛苦": (-0.9, 0.4, -0.6),
    "绝望": (-0.9, 0.3, -0.75),
    "失望": (-0.75, -0.25, -0.45),
    "沮丧": (-0.7, -0.4, -0.5),
    "郁闷": (-0.65, -0.3, -0.4),
    "生气": (-0.85, 0.75, 0.5),
    "愤怒": (-0.9, 0.8, 0.55),
    "恼火": (-0.8, 0.7, 0.4),
    "害怕": (-0.75, 0.65, -0.6),
    "恐惧": (-0.85, 0.75, -0.7),
    "担心": (-0.55, 0.45, -0.35),
    "焦虑": (-0.55, 0.65, -0.45),
    "紧张": (-0.5, 0.7, -0.3),
    "恶心": (-0.85, -0.2, -0.4),
    "讨厌": (-0.75, 0.2, -0.35),
    "厌恶": (-0.8, -0.1, -0.45),
    "嫌弃": (-0.7, -0.15, -0.3),
    "后悔": (-0.7, -0.15, -0.45),
    "内疚": (-0.55, 0.25, -0.45),
    "惭愧": (-0.5, 0.15, -0.4),
    "尴尬": (-0.45, 0.35, -0.25),
    "羞愧": (-0.6, 0.3, -0.35),
    "羡慕": (-0.35, 0.35, -0.2),
    "嫉妒": (-0.45, 0.4, -0.15),
    "失落": (-0.6, -0.3, -0.4),
    "孤独": (-0.65, -0.4, -0.5),
    "寂寞": (-0.6, -0.35, -0.45),
    "无奈": (-0.55, -0.2, -0.35),
    "无聊": (-0.3, -0.4, 0.0),

    # 中性/高唤醒
    "惊讶": (0.1, 0.85, 0.2),
    "意外": (0.0, 0.7, 0.15),
    "震惊": (-0.3, 0.85, -0.1),
    "期待": (0.45, 0.55, 0.5),
    "希望": (0.55, 0.5, 0.55),
    "激动": (0.75, 0.85, 0.55),
}


def get_emotion_by_name(name: str) -> Optional[EmotionDefinition]:
    """根据名称获取情感定义"""
    return EMOTION_DEFINITIONS.get(name.lower())


def get_primary_emotions() -> List[str]:
    """获取所有原始情感名称"""
    return ["joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust"]


def get_complex_emotions() -> List[str]:
    """获取所有复合情感名称"""
    return list(EMOTION_DEFINITIONS.keys() - set(get_primary_emotions()))


def get_all_emotions() -> List[str]:
    """获取所有情感名称"""
    return list(EMOTION_DEFINITIONS.keys())


def vad_to_emotion_guess(v: float, a: float, d: float) -> str:
    """
    根据VAD值猜测对应情感（辅助函数）
    基于Russell的环形情感模型
    """
    # 高唤醒
    if a > 0.5:
        if v > 0.5:
            return "joy"
        elif v < -0.5:
            return "anger"
        else:
            return "surprise"
    # 低唤醒
    else:
        if v > 0.5:
            return "contentment"
        elif v < -0.5:
            return "sadness"
        else:
            return "neutral"


def is_primary_emotion(name: str) -> bool:
    """判断是否为原始情感"""
    return name.lower() in get_primary_emotions()


def is_complex_emotion(name: str) -> bool:
    """判断是否为复合情感"""
    return name.lower() in get_complex_emotions()
