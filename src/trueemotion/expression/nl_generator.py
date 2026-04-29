# -*- coding: utf-8 -*-
"""
NaturalLanguageGenerator - 自然语言生成器

将情感意图转化为自然、口语化的中文表达。
不是模板填充，而是基于规则的灵活生成。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random
import re


@dataclass
class NLGConfig:
    """NLG配置"""
    # 口语化程度 0.0-1.0
    colloquial_level: float = 0.7

    # 随机性程度 0.0-1.0（增加输出的多样性）
    randomness: float = 0.3

    # 平均句子长度
    avg_sentence_length: float = 15.0

    # 是否添加语气词
    use_particles: bool = True

    # 是否使用标点强调
    use_punctuation_emphasis: bool = True


class NaturalLanguageGenerator:
    """
    自然语言生成器

    核心功能：
    1. 口语化转换 - 书面语 → 口语
    2. 语气词注入 - 添加"啊"、"呗"、"嘛"等
    3. 句子多样化 - 避免重复的句式
    4. 标点强调 - 感叹号、省略号的使用
    5. 随机性 - 同一内容多种表达

    使用示例:
        nlg = NaturalLanguageGenerator()
        text = nlg.generate(
            intent="我理解你，真的",
            emotion="sadness",
            intensity=0.8
        )
    """

    # 书面语 → 口语 转换映射
    FORMAL_TO_CASUAL = {
        "非常感谢": ["谢谢啊", "谢啦", "感谢", "谢咯"],
        "非常": ["特别", "贼", "超", "巨", "忒"],
        "特别": ["特别", "老", "贼", "超"],
        "我认为": ["我觉着", "我觉得", "我寻思", "我寻思着"],
        "但是": ["不过", "但", "只是"],
        "因此": ["所以", "于是", "这不"],
        "所以": ["所以", "于是", "这不"],
        "而且": ["而且", "还", "并且"],
        "如果": ["要是", "假如", "要是说"],
        "虽然": ["虽说", "虽然", "就", "虽然说"],
        "因为": ["因为", "由于", "这不"],
        "可以": ["能", "可以", "中"],
        "不知道": ["不知道", "不清楚", "俺也不造"],
        "怎么样": ["咋样", "怎么样", "啥情况"],
        "什么": ["啥", "什么", "嘛"],
        "为什么": ["为啥", "为啥", "咋回事"],
        "非常感谢": ["谢谢啊", "谢啦", "感谢感谢"],
        "理解你的感受": ["懂你", "懂你的感受", "理解"],
        "感同身受": ["感同身受", "换我我也", "我懂"],
        "太棒了": ["太牛了", "太厉害了", "绝了", "厉害"],
        "太好了": ["太好了", "太好了", "蛮好的", "不错"],
        "我知道了": ["知道了", "懂了", "明白", "了解"],
        "请注意": ["注意", "小心", "哎", "诶"],
        "综上所述": ["反正", "总之", "就是说"],
        "然而": ["但", "不过", "可", "然而"],
        "此外": ["另外", "还有", "而且"],
        "总的来说": ["反正", "总之", "总的来看"],
    }

    # 语气词列表（按风格分类）
    PARTICLES = {
        "soft": ["呗", "嘛", "呀", "哦", "呢"],
        "enthusiastic": ["啊", "哈", "哇", "呀", "啦"],
        "restrained": ["呃", "嗯", "这个", "那个"],
        "question": ["吗", "呢", "吧", "呀"],
    "neutral": ["嗯", "哦", "嘛", "哔"],
    }

    # 句子前缀（用于开头）
    SENTENCE_STARTERS = [
        "说实话",
        "说真的",
        "其实",
        "你看",
        "你知道吗",
        "我就说",
        "坦白讲",
        "跟你讲",
        "老实说",
        "不瞒你说",
    ]

    # 填充词（用于句中）
    FILLERS = [
        "呃",
        "那个",
        "嗯",
        "就是说",
        "其实吧",
        "怎么说呢",
        "你懂我意思吧",
        "这样",
        "然后",
        "所以",
    ]

    # 感叹词
    INTERJECTIONS = {
        "positive": ["哇", "哇塞", "好家伙", "我的天", "牛啊", "厉害", "绝了"],
        "negative": ["哎", "唉", "呃", "晕", "好家伙"],
        "surprise": ["哇", "我天", "真的假的", "不会吧", "真的假的"],
        "neutral": ["嗯", "哦", "这样啊", "是嘛"],
    }

    def __init__(self, config: Optional[NLGConfig] = None, seed: Optional[int] = None):
        self.config = config or NLGConfig()
        self.rng = random.Random(seed)

    def generate(
        self,
        intent: str,
        emotion: str = "neutral",
        intensity: float = 0.5,
        personality: Optional[Dict] = None,
    ) -> str:
        """
        生成自然语言

        Args:
            intent: 核心意图文本
            emotion: 当前情感
            intensity: 情感强度
            personality: 个性特征（可选）

        Returns:
            自然语言文本
        """
        result = intent

        # 1. 口语化转换
        result = self._casualize(result)

        # 2. 添加语气词
        if self.config.use_particles:
            result = self._add_particles(result, emotion, intensity)

        # 3. 添加标点强调
        if self.config.use_punctuation_emphasis:
            result = self._apply_punctuation(result, emotion, intensity)

        # 4. 句子多样化
        result = self._vary_sentence(result, emotion)

        # 5. 可能添加开场白
        if self.rng.random() < self.config.randomness * 0.5:
            result = self._add_opening(result, emotion)

        return result

    def _casualize(self, text: str) -> str:
        """将书面语转换为口语"""
        # 按长度降序排列 key，避免部分替换问题
        sorted_keys = sorted(self.FORMAL_TO_CASUAL.keys(), key=len, reverse=True)

        for formal in sorted_keys:
            if formal in text:
                casual_options = self.FORMAL_TO_CASUAL[formal]
                casual = self.rng.choice(casual_options)
                text = text.replace(formal, casual)

        return text

    def _add_particles(self, text: str, emotion: str, intensity: float) -> str:
        """添加语气词"""
        if len(text) < 3:
            return text

        # 根据情感选择语气词类型
        if emotion in ["joy", "surprise", "excitement"]:
            particle_type = "enthusiastic"
        elif emotion in ["sadness", "fear", "anxiety"]:
            particle_type = "soft"
        elif emotion in ["anger"]:
            particle_type = "restrained"  # 生气时反而克制
        else:
            particle_type = "neutral"

        particles = self.PARTICLES[particle_type]

        # 句尾添加语气词
        if not text[-1] in "。！？，":
            if self.rng.random() < 0.6:
                particle = self.rng.choice(particles)
                # 避免重复
                if particle not in text[-2:]:
                    text = text + particle

        # 高强度情感添加更多语气词
        if intensity > 0.7 and self.rng.random() < 0.4:
            filler = self.rng.choice(self.FILLERS)
            if filler not in text[:5]:  # 不重复开头
                text = text[:3] + filler + text[3:]

        return text

    def _apply_punctuation(self, text: str, emotion: str, intensity: float) -> str:
        """应用标点强调"""
        # 高强度正面情感用感叹号
        if intensity > 0.7 and emotion in ["joy", "surprise", "love", "excitement"]:
            if not any(p in text for p in "！？"):
                # 感叹号替代句号
                text = text.rstrip("。") + "！"
            elif text.endswith("。"):
                text = text[:-1] + "！"

        # 悲伤情感用省略号
        elif intensity > 0.7 and emotion in ["sadness", "despair", "grief"]:
            if "。" in text and "..." not in text:
                text = text.replace("。", "...")

        # 愤怒情感用感叹号
        elif intensity > 0.6 and emotion in ["anger", "contempt"]:
            if "。" in text:
                text = text.replace("。", "！")

        return text

    def _vary_sentence(self, text: str, emotion: str) -> str:
        """增加句子多样性"""
        # 随机决定是否变换句式
        if self.rng.random() > self.config.randomness:
            return text

        # 拆分句子
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return text

        # 随机交换句子顺序（保留最后一句）
        if len(sentences) > 2 and self.rng.random() < 0.3:
            middle = sentences[1:-1]
            self.rng.shuffle(middle)
            sentences = [sentences[0]] + middle + [sentences[-1]]

        # 添加连接词
        connectors = ["然后", "接着", "再说", "而且", "还有"]
        for i in range(len(sentences) - 1):
            if self.rng.random() < 0.3:
                connector = self.rng.choice(connectors)
                sentences[i] = sentences[i] + "，" + connector

        # 重新组合
        result = "。".join(sentences)
        if not any(p in result for p in "！？"):
            result += "。"

        return result

    def _add_opening(self, text: str, emotion: str) -> str:
        """添加开场白"""
        # 情感化的开场白
        if emotion in ["joy", "surprise"]:
            openings = ["说真的", "其实", "你看", "你知道吗"]
        elif emotion in ["sadness", "despair"]:
            openings = ["其实", "说真的", "你看"]
        elif emotion in ["anger"]:
            openings = ["说实话", "坦白讲", "说真的"]
        else:
            openings = self.SENTENCE_STARTERS[:5]

        if self.rng.random() < 0.3:
            opening = self.rng.choice(openings)
            if opening not in text[:5]:  # 不重复
                text = opening + "，" + text

        return text

    def generate_variants(
        self, intent: str, emotion: str, intensity: float, n: int = 3
    ) -> List[str]:
        """生成n个变体，用于选择或测试"""
        variants = set()
        while len(variants) < n:
            variant = self.generate(intent, emotion, intensity)
            variants.add(variant)
        return list(variants)

    def add_humor(
        self, text: str, humor_level: float = 0.5
    ) -> str:
        """为文本添加幽默元素"""
        if humor_level < 0.3:
            return text

        humor_additions = [
            "哈哈",
            "😄",
            "好家伙",
            "笑死我了",
            "绝了",
        ]

        if self.rng.random() < humor_level:
            addition = self.rng.choice(humor_additions[:3])
            if addition not in text:
                text = f"{text} {addition}"

        return text

    def soften(self, text: str) -> str:
        """软化表达（用于敏感话题）"""
        softeners = [
            ("死了", "够呛"),
            ("讨厌", "不太喜欢"),
            ("恨", "不太喜欢"),
            ("很差", "不太行"),
            ("垃圾", "不太行"),
            ("傻", "不太聪明"),
            ("笨", "需要多学习"),
        ]

        for hard, soft in softeners:
            if hard in text:
                text = text.replace(hard, soft)

        return text

    def intensify(self, text: str) -> str:
        """强化表达（用于需要强调时）"""
        intensifiers = ["超", "特别", "非常", "贼", "巨", "忒"]
        # 在形容词前添加
        for adv in intensifiers:
            if self.rng.random() < 0.4:
                # 找到形容词位置
                adjectives = ["好", "棒", "不错", "厉害", "牛"]
                for adj in adjectives:
                    if adj in text and adv + adj not in text:
                        text = text.replace(adj, adv + adj)
                        break

        return text


@dataclass
class DialogueAct:
    """对话行为类型"""
    GREETING = "greeting"
    EMPATHY = "empathy"
    QUESTION = "question"
    STATEMENT = "statement"
    ANSWER = "answer"
    FAREWELL = "farewell"


class DialogueNLG(NaturalLanguageGenerator):
    """
    对话型自然语言生成器

    在基础NLG上增加了对话行为支持：
    - 问句生成
    - 回答生成
    - 结束语生成
    """

    # 问句模板
    QUESTION_TEMPLATES = {
        "clarifying": [
            "你的意思是...？",
            "具体是什么情况？",
            "能详细说说吗？",
        ],
        "empathetic": [
            "你当时怎么想的？",
            "你有什么感受？",
            "后来发生了什么？",
        ],
        "action": [
            "你觉得怎么办好？",
            "要不要试试...？",
            "有什么我能帮你的？",
        ],
    }

    # 结束语模板
    FAREWELL_TEMPLATES = [
        "先这样，有事儿再说",
        "回头聊",
        "加油！",
        "晚安",
        "拜拜～",
    ]

    def generate_question(
        self, question_type: str = "clarifying", emotion: str = "neutral"
    ) -> str:
        """生成问句"""
        templates = self.QUESTION_TEMPLATES.get(
            question_type,
            self.QUESTION_TEMPLATES["clarifying"]
        )
        question = self.rng.choice(templates)
        return self.generate(question, emotion)

    def generate_farewell(self) -> str:
        """生成结束语"""
        farewell = self.rng.choice(self.FAREWELL_TEMPLATES)
        return self.generate(farewell, emotion="neutral")


if __name__ == "__main__":
    # 测试
    nlg = NaturalLanguageGenerator(
        config=NLGConfig(colloquial_level=0.8, randomness=0.4),
        seed=42
    )

    print("=== Natural Language Generator Test ===\n")

    test_cases = [
        ("我理解你的感受", "sadness", 0.8),
        ("太棒了！恭喜你！", "joy", 0.95),
        ("这确实让人很生气", "anger", 0.7),
        ("非常感谢你的帮助", "neutral", 0.3),
        ("我觉得你做得很好", "joy", 0.6),
    ]

    for intent, emotion, intensity in test_cases:
        result = nlg.generate(intent, emotion, intensity)
        print(f"Input: {intent}")
        print(f"  ({emotion}, {intensity})")
        print(f"Output: {result}")
        print()

    # 生成多个变体
    print("=== Multiple Variants ===")
    variants = nlg.generate_variants("我理解你的感受", "sadness", 0.7, n=3)
    for i, v in enumerate(variants):
        print(f"  Variant {i+1}: {v}")