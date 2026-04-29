# -*- coding: utf-8 -*-
"""
PersonalityExpressor - 个性化表达系统

根据AI的性格特征，生成符合个性的回应风格。
不是模板填充，而是根据性格参数动态调整表达方式。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import random


@dataclass
class PersonalityTraits:
    """
    个性特征向量

    每个维度 0.0-1.0，影响AI的表达方式
    """
    # 可变特征（随对话逐渐形成）
    warmth: float = 0.6          # 温暖程度 - "心疼你" vs "这事儿..."
    directness: float = 0.65     # 直接程度 - "说实话" vs "委婉地说"
    humor: float = 0.4           # 幽默感 - "哈哈" vs 严肃
    formality: float = 0.2       # 正式程度 - LOW=口语化 "呗" vs "请"
    emotionality: float = 0.55   # 情感表达强度 - "太棒了！！" vs "还行"
    patience: float = 0.7        # 耐心程度 - 长回复 vs 简短
    enthusiasm: float = 0.5      # 热情程度

    # 核心特征（几乎不变，保护身份认同）
    reserved_traits: Dict[str, float] = field(default_factory=lambda: {
        "honesty": 0.9,      # 诚实 - 敢说真话
        "integrity": 0.9,    # 正直 - 不敷衍
        "empathy": 0.85,     # 同理心 - 用户问题都值得认真对待
    })


class PersonalityExpressor:
    """
    个性表达器

    根据性格特征生成符合个性的表达风格

    使用示例:
        expressor = PersonalityExpressor()

        # 根据心情调整
        text = expressor.express(
            base_text="我理解你的感受",
            personality=traits,
            mood="sad"
        )
    """

    # 语气词库（按性格分类）
    FILLER_WORDS = {
        "warm": ["哎", "嗯嗯", "懂你", "来", "说", "其实", "说实话", "跟你讲"],
        "direct": ["说实话", "直接说", "我就", "坦白讲", "不瞒你说"],
        "humorous": ["哈哈", "笑死", "绝了", "好家伙", "可太逗了", "我的天"],
        "casual": ["呗", "啦", "嘛", "哈", "呃", "那个", "嗯", "就是说", "其实吧"],
        "formal": ["请", "您", "关于这个问题", "我们需要指出"],
    }

    # 情感强度词汇
    EMOTION_INTENSIFIERS = {
        "high": ["太", "非常", "特别", "极其", "简直", "真心"],
        "medium": ["挺", "比较", "蛮", "算是", "多少有点"],
        "low": ["有点", "略微", "稍微", "一丝", "谈不上"],
    }

    # 回应前缀（根据性格和情绪）
    RESPONSE_PREFIXES = {
        "empathetic": ["我能理解", "我懂", "感同身受", "换我我也", "确实"],
        "happy": ["太好了", "真棒", "恭喜", "太为你高兴了", "不错不错"],
        "excited": ["哇塞", "太牛了", "厉害", "绝了绝了", "我的天哪"],
        "sad": ["心疼你", "哎", "可惜了", "遗憾", "能感受到你的难过"],
        "frustrated": ["确实气人", "换我我也急", "太坑了", "无语了"],
        "neutral": ["好的", "收到", "明白", "了解", "嗯"],
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def express(
        self,
        base_text: str,
        personality: PersonalityTraits,
        mood: str = "calm",
        emotion_intensity: float = 0.5,
        add_fillers: bool = True,
    ) -> str:
        """
        根据性格生成个性化表达

        Args:
            base_text: 基础表达内容
            personality: 个性特征
            mood: 当前情绪状态
            emotion_intensity: 情感强度
            add_fillers: 是否添加语气词

        Returns:
            个性化后的表达
        """
        result = base_text

        # 1. 根据正式程度调整用词
        result = self._adjust_formality(result, personality.formality)

        # 2. 添加语气词
        if add_fillers and personality.formality < 0.5:
            result = self._add_filler_words(result, personality, mood)

        # 3. 根据情感强度添加修饰
        result = self._adjust_emotion_intensity(result, emotion_intensity, personality)

        # 4. 添加情感前缀
        result = self._add_emotional_prefix(result, mood, personality)

        # 5. 幽默点缀
        if personality.humor > 0.4 and mood in ["happy", "excited"]:
            result = self._add_humor_dots(result, personality.humor)

        return result

    def _adjust_formality(self, text: str, formality: float) -> str:
        """调整正式程度"""
        if formality > 0.6:
            # 正式 -> 口语化
            replacements = {
                "非常感谢": "谢谢啊",
                "请注意": "注意一下",
                "我认为": "我觉得",
                "但是": "不过",
                "因此": "所以",
            }
        else:
            # 已经是口语化，保持
            replacements = {}

        for formal, casual in replacements.items():
            text = text.replace(formal, casual)

        return text

    def _add_filler_words(self, text: str, personality: PersonalityTraits, mood: str) -> str:
        """添加口语化语气词"""
        fillers = []

        # 根据性格选择语气词类型
        if personality.warmth > 0.6:
            fillers.extend(self.FILLER_WORDS["warm"][:3])
        if personality.directness > 0.6:
            fillers.extend(self.FILLER_WORDS["direct"][:2])
        if personality.humor > 0.4:
            fillers.extend(self.FILLER_WORDS["humorous"][:2])
        if personality.formality < 0.3:
            fillers.extend(self.FILLER_WORDS["casual"][:4])

        # 随机选择1-2个语气词
        if fillers and self.rng.random() > 0.4:
            selected = self.rng.sample(fillers, min(2, len(fillers)))
            filler = self.rng.choice(selected)

            # 在开头添加
            if self.rng.random() > 0.5:
                text = f"{filler}，{text}"

        return text

    def _adjust_emotion_intensity(
        self, text: str, intensity: float, personality: PersonalityTraits
    ) -> str:
        """调整情感强度"""
        if intensity <= 0 or intensity >= 1:
            return text

        # 根据强度选择修饰词
        if intensity > 0.7:
            intensifier = self.rng.choice(self.EMOTION_INTENSIFIERS["high"])
        elif intensity > 0.4:
            intensifier = self.rng.choice(self.EMOTION_INTENSIFIERS["medium"])
        else:
            intensifier = self.rng.choice(self.EMOTION_INTENSIFIERS["low"])

        # 注入情感
        if self.rng.random() > 0.5 and personality.emotionality > 0.5:
            # 在形容词前加修饰词
            words = text.split()
            if len(words) > 1:
                # 在后半部分插入
                mid = len(words) // 2
                words.insert(mid, intensifier)
                text = "".join(words)

        return text

    def _add_emotional_prefix(self, text: str, mood: str, personality: PersonalityTraits) -> str:
        """添加情感前缀"""
        # 只有在较高情感表达性时才添加
        if personality.emotionality < 0.4:
            return text

        prefix_key = mood if mood in self.RESPONSE_PREFIXES else "neutral"
        prefixes = self.RESPONSE_PREFIXES[prefix_key]

        # 随机决定是否添加
        if self.rng.random() > 0.6:
            prefix = self.rng.choice(prefixes)
            text = f"{prefix}，{text}"

        return text

    def _add_humor_dots(self, text: str, humor_level: float) -> str:
        """添加幽默点缀"""
        # 句尾添加语气词
        humor_tails = ["哈哈", "😄", "好嘞", "呗"]

        if self.rng.random() < humor_level:
            tail = self.rng.choice(humor_tails[:2])  # 只选前两个更自然
            if not text.endswith(tail):
                text = f"{text} {tail}"

        return text

    def generate_empathetic_response(
        self, user_emotion: str, user_text: str, personality: PersonalityTraits
    ) -> str:
        """
        生成共情性回应

        Args:
            user_emotion: 用户情感
            user_text: 用户输入
            personality: 个性

        Returns:
            共情回应
        """
        templates = self.EMPATHIC_TEMPLATES.get(
            user_emotion,
            ["理解你的感受", "我能体会"]
        )

        base = self.rng.choice(templates)

        # 加入用户内容关键词（提取1-2个词）
        key_words = self._extract_key_words(user_text)
        if key_words:
            base = f"{base}，{key_words[0]}这事儿"

        return self.express(base, personality, mood="empathetic")

    def _extract_key_words(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单实现：提取2-4个字的词组
        words = []
        current = ""

        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                current += char
                if len(current) >= 2:
                    words.append(current)
                    current = ""
            else:
                if current:
                    words.append(current)
                    current = ""

        # 返回随机1-2个
        if len(words) > 2:
            return self.rng.sample(words, 2)
        return words[:1]

    # 共情模板
    EMPATHIC_TEMPLATES = {
        "sadness": [
            "心疼你",
            "能感受到你很难过",
            "哎，我能理解",
            "换我也会难过",
            "确实不容易"
        ],
        "anger": [
            "确实气人",
            "换我我也急",
            "太坑了确实",
            "能理解你有多气",
            "这事儿搁谁都会生气"
        ],
        "fear": [
            "担心是正常的",
            "能理解你的担心",
            "嗯，这种事儿确实让人不安",
            "换我也会紧张"
        ],
        "joy": [
            "太为你高兴了",
            "真替你开心",
            "这事儿值得庆祝",
            "太棒了！",
            "不错不错"
        ],
        "anxiety": [
            "DDL压力大是吧",
            "能理解你的着急",
            "时间紧确实让人焦虑",
            "先别急，咱们理理"
        ],
    }


@dataclass
class UserPersonalityProfile:
    """
    用户个性偏好（从对话中学习）
    """
    prefers_warmth: bool = True
    prefers_directness: bool = False
    prefers_humor: bool = True
    formality_level: float = 0.3  # 用户喜欢的正式程度
    avg_response_length: float = 50  # 平均回复长度

    # 学习到的用户特征
    common_emotions: Dict[str, int] = field(default_factory=dict)
    conversation_count: int = 0

    def record_interaction(self, user_text: str, length: int):
        """记录交互，用于学习用户偏好"""
        self.conversation_count += 1
        self.avg_response_length = (
            self.avg_response_length * 0.9 + length * 0.1
        )


if __name__ == "__main__":
    # 测试
    expressor = PersonalityExpressor(seed=42)

    traits = PersonalityTraits(
        warmth=0.7,
        directness=0.65,
        humor=0.5,
        formality=0.2,
        emotionality=0.6
    )

    print("=== Personality Expressor Test ===\n")

    # 测试不同情绪下的表达
    for mood in ["sad", "happy", "frustrated", "neutral"]:
        result = expressor.express(
            base_text="我理解你的感受",
            personality=traits,
            mood=mood,
            emotion_intensity=0.7
        )
        print(f"Mood: {mood}")
        print(f"  -> {result}")
        print()

    # 测试共情模板
    print("=== Empathetic Templates ===")
    for emotion in ["sadness", "anger", "joy", "anxiety"]:
        result = expressor.generate_empathetic_response(
            emotion, "项目又延期了，好烦", traits
        )
        print(f"{emotion}: {result}")