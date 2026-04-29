# -*- coding: utf-8 -*-
"""
Murakami Haruki Style Emotional Text Generator
村上春树风格的情感文本生成器

特点：
- 间接表达，不过度情绪化
- 大量隐喻和象征
- 细腻的心理描写
- 留白艺术
- 爵士乐和猫等意象
"""

import random
import numpy as np
from typing import Dict, List, Tuple


# ==================== 村上春树风格的意象词库 ====================

MURAKAMI_OBJECTS = {
    'positive': [
        '猫在窗台上晒太阳', '爵士乐在房间里流淌', '威士忌的琥珀色光泽',
        '书架上的灰尘在阳光下跳舞', '凌晨四点的沉默', '旧唱片转动的节奏',
        '冰箱里最后一块冰块', '阳光照进昏暗的房间', '咖啡的香气弥漫',
        '打字机清脆的声音', '火车经过时的震动', '雨后的空气',
    ],
    'negative': [
        '冰箱坏了里面的东西在慢慢腐烂', '凌晨三点无法入眠',
        '抽屉里找不到的东西', '过期未回的信', '墙上停止的挂钟',
        '逐渐模糊的记忆', '无法到达的列车时刻表', '淋湿的火柴',
        '阳台枯萎的盆栽', '再也听不到的那首歌', '卡住的唱片针',
    ],
    'neutral': [
        '窗外的雨滴声', '书架上排列整齐的书脊', '桌上的台灯',
        '墙上的画框', '地上的影子', '天空的颜色',
        '时间的流逝', '沉默的对话', '记忆的碎片',
    ],
}

# 情感连接词 - 村上式的转折
MURAKAMI_CONNECTORS = [
    '然而', '只是', '不过', '说起来', '话虽这么说',
    '总之', '然后', '于是', '某种程度上',
    '某种意义来说', '某种程度上', '或者说',
]

# 情感动词 - 含蓄的表达
MURAKAMI_VERBS = {
    'joy': [
        '某种东西在胸口轻轻散开', '感觉到什么在融化',
        '有什么轻轻地落入正确的位置', '像是被什么轻轻托起',
        '某种重量突然消失了', '有什么在体内慢慢苏醒',
        '像是喝了一杯温度刚好的咖啡',
    ],
    'sadness': [
        '某种东西在胸口慢慢下沉', '感觉到什么在逐渐凝固',
        '有什么在慢慢缺失', '像是被什么轻轻放下',
        '某种重量压在肩上', '有什么在体内慢慢变冷',
        '像是窗外渐渐暗下去的天色',
    ],
    'anger': [
        '某种东西在胸口慢慢升温', '感觉到什么在体内翻涌',
        '有什么想要冲破什么', '像是被堵住的水管',
        '某种压力在慢慢积累', '有什么在体内慢慢燃烧',
        '像是咖啡杯里越来越深的漩涡',
    ],
    'anxiety': [
        '某种东西在胃里纠结', '感觉到什么在不安地等待',
        '有什么在体内轻轻震颤', '像是深夜的电话铃声',
        '某种预感在慢慢成形', '有什么在体内轻轻收缩',
        '像是即将到来的列车声',
    ],
    'love': [
        '某种东西在记忆里轻轻浮现', '感觉到什么在远处闪烁',
        '有什么在体内某个角落静静存在', '像是褪色的照片',
        '某种温度在慢慢传递', '有什么在体内某个地方柔软下来',
        '像是旧唱片里某段熟悉的旋律',
    ],
    'loneliness': [
        '某种东西在房间里慢慢弥漫', '感觉到什么在四壁之间回响',
        '有什么在体内某个地方空着', '像是缺少了什么的声音',
        '某种沉默在慢慢扩散', '有什么在体内某个地方凝固',
        '像是凌晨空荡荡的便利店',
    ],
    'confusion': [
        '某种东西在脑海里漂浮', '感觉到什么在雾中若隐若现',
        '有什么在记忆的边缘徘徊', '像是找不到出口的走廊',
        '某种模糊在慢慢成形', '有什么在体内某个地方错位',
        '像是书架上放错位置的那本书',
    ],
}

# 身体感受词 - 村上式的身体语言
MURAKAMI_BODY = [
    '胸口有什么在轻轻震动', '指尖传来某种温度',
    '喉咙深处有什么在收紧', '胃里有什么在慢慢下沉',
    '后颈感到某种重量', '太阳穴在轻轻跳动',
    '后背有某种触感在蔓延', '眼睛后面有什么在闪烁',
]

# 时间表达 - 村上的时间感
MURAKAMI_TIME = [
    '凌晨四点', '傍晚六点', '深夜十一点',
    '雨停了之后的黄昏', '最后一班地铁', '午夜的便利店',
    '黎明前的黑暗', '咖啡冷掉之前', '唱片转完之前',
]


class MurakamiEmotionGenerator:
    """村上春树风格情感文本生成器"""

    EMOTION_NAMES = [
        "joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust",
        "optimism", "love", "guilt", "submission", "surprise_complex", "disappointment",
        "remorse", "envy", "suspicion", "aggression", "pride", "contentment", "contempt",
        "cynicism", "morbidness", "sentimentality", "anxiety", "despair"
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.emotion_to_idx = {e: i for i, e in enumerate(self.EMOTION_NAMES)}

    def _build_sentence(self, emotion: str) -> str:
        """构建村上春树风格的句子"""

        templates = []

        # 类型1: 意象 + 情感动词
        templates.append(lambda: random.choice(MURAKAMI_OBJECTS['positive' if random.random() < 0.5 else 'negative']) +
                        '，' + random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])))

        # 类型2: 时间 + 身体感受
        templates.append(lambda: random.choice(MURAKAMI_TIME) +
                        '，' + random.choice(MURAKAMI_BODY))

        # 类型3: 意象 + 转折 + 情感动词
        templates.append(lambda: random.choice(MURAKAMI_OBJECTS['neutral']) +
                        '，' + random.choice(MURAKAMI_CONNECTORS) +
                        '，' + random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])))

        # 类型4: 简短描述 + 留白
        templates.append(lambda: random.choice(MURAKAMI_OBJECTS['neutral']) +
                        '。' + random.choice(MURAKAMI_TIME) +
                        '。' + random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])) +
                        '。')

        # 类型5: 自我对话式
        templates.append(lambda: '窗外的' + ('雨声' if random.random() < 0.5 else '阳光') +
                        '让我想起什么。' +
                        random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])) +
                        '，某种程度上。')

        # 类型6: 爵士乐/音乐式
        templates.append(lambda: '唱片转动的节奏里，' +
                        random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])) +
                        '，像是某段被遗忘的旋律。')

        # 类型7: 猫式（村上最爱）
        templates.append(lambda: '猫在沙发上睡着。' +
                        random.choice(MURAKAMI_TIME) +
                        '，我在想一些事情。' +
                        random.choice(MURAKAMI_VERBS.get(emotion, MURAKAMI_VERBS['confusion'])) +
                        '。')

        template = random.choice(templates)
        return template()

    def _generate_vad(self, emotion: str) -> Tuple[float, float, float]:
        """生成VAD标签"""
        vad_map = {
            'joy': (0.70, 0.40, 0.60),
            'sadness': (-0.70, -0.20, -0.40),
            'anger': (-0.60, 0.50, 0.30),
            'anxiety': (-0.40, 0.50, -0.30),
            'loneliness': (-0.50, -0.10, -0.30),
            'love': (0.75, 0.30, 0.65),
            'confusion': (-0.20, 0.30, -0.20),
            'fear': (-0.50, 0.45, -0.35),
            'despair': (-0.75, 0.20, -0.55),
            'contentment': (0.60, 0.20, 0.55),
            'remorse': (-0.55, 0.15, -0.40),
            'longing': (0.40, 0.50, 0.30),
        }
        base = vad_map.get(emotion, (0, 0, 0))
        vad = tuple(np.clip(base[i] + np.random.randn() * 0.08, -1, 1) for i in range(3))
        return vad

    def generate_single(self, emotion: str = None) -> Dict:
        """生成单条样本"""
        if emotion is None:
            emotion = random.choice(list(MURAKAMI_VERBS.keys()))

        text = self._build_sentence(emotion)

        return {
            "text": text,
            "emotion": emotion,
            "vad": self._generate_vad(emotion),
            "intensity": random.uniform(0.4, 0.75),  # 村上风格偏含蓄
        }

    def generate_batch(self, num_samples: int) -> Dict:
        """生成批量样本"""
        texts = []
        emotion_idxs = []
        vad_labels = []
        intensity_labels = []

        emotions = list(MURAKAMI_VERBS.keys()) + ['contentment', 'remorse', 'longing']

        for _ in range(num_samples):
            emotion = random.choice(emotions)
            sample = self.generate_single(emotion)
            if sample:
                texts.append(sample["text"])
                emotion_idxs.append(self.emotion_to_idx.get(emotion, 0))
                vad_labels.append(sample["vad"])
                intensity_labels.append(sample["intensity"])

        return {
            "texts": texts,
            "emotion_idxs": np.array(emotion_idxs),
            "vad_labels": np.array(vad_labels, dtype=np.float32),
            "intensity_labels": np.array(intensity_labels, dtype=np.float32),
        }


# ==================== 扩展：更多含蓄表达 ====================

RESERVED_EXPRESSIONS = {
    # 中国古典诗词风格的含蓄表达
    'chinese_poetry': [
        '庭院深深深几许，杨柳堆烟，帘幕无重数',
        '泪眼问花花不语，乱红飞过秋千去',
        '人生若只如初见，何事秋风悲画扇',
        '此情无计可消除，才下眉头，却上心头',
        '众里寻他千百度，蓦然回首，那人却在灯火阑珊处',
        '问君能有几多愁，恰似一江春水向东流',
        '衣带渐宽终不悔，为伊消得人憔悴',
        '昨夜西风凋碧树，独上高楼，望尽天涯路',
    ],
    # 日本文学的物哀风格
    'mono_no_aware': [
        '樱花飘落的瞬间，美得让人心碎',
        '秋日的夕阳，像是谁的叹息',
        '雪融化的声音，像是某种告别',
        '月亮被云遮住，世界陷入短暂的黑暗',
        '落叶归根，像是无声的归宿',
        '潮起潮落，像是无言的呼吸',
        '风吹过竹林的声音，像是遥远的回忆',
        '雨后的彩虹，美得让人想要流泪',
    ],
    # 现代含蓄表达
    'modern_reserved': [
        '有些话到嘴边又咽了回去',
        '沉默比任何语言都更响亮',
        '笑容背后藏着什么',
        '眼神里有什么一闪而过',
        '手指在桌面上轻轻敲打',
        '窗外的风景在变换，心里的什么也在改变',
        '有些东西说不清楚，但能感觉到',
        '漫长的等待，无声的期盼',
    ],
}


class ReservedEmotionGenerator:
    """含蓄情感文本生成器"""

    EMOTION_NAMES = MurakamiEmotionGenerator.EMOTION_NAMES

    def __init__(self, seed: int = 43):
        random.seed(seed)
        np.random.seed(seed)
        self.emotion_to_idx = {e: i for i, e in enumerate(self.EMOTION_NAMES)}

    def generate_single(self, emotion: str = None) -> Dict:
        """生成单条样本"""
        if emotion is None:
            sources = ['chinese_poetry', 'mono_no_aware', 'modern_reserved']
            source = random.choice(sources)
            texts = RESERVED_EXPRESSIONS[source]
            text = random.choice(texts)
        else:
            # 根据情感选择合适的含蓄表达
            text = self._get_reserved_text(emotion)

        vad = self._generate_vad(emotion)

        return {
            "text": text,
            "emotion": emotion or 'confusion',
            "vad": vad,
            "intensity": random.uniform(0.3, 0.65),
        }

    def _get_reserved_text(self, emotion: str) -> str:
        """根据情感获取含蓄文本"""
        emotion_texts = {
            'sadness': [
                '有些东西失去了就再也回不来',
                '沉默是今晚的康桥',
                '有些故事没有结局',
                '窗外的雨像是心里的泪',
            ],
            'joy': [
                '阳光正好，风也温柔',
                '某些时刻，时间仿佛停止了',
                '简单的日子里藏着小小的幸福',
                '风吹过树叶的沙沙声，像是某种回应',
            ],
            'loneliness': [
                '空荡荡的房间里，只有影子作伴',
                '人群中的孤独，更显得刺眼',
                '有些路注定要一个人走',
                '深夜的便利店，空旷的货架',
            ],
            'love': [
                '有些话不用说，心里都懂',
                '目光交汇的瞬间，千言万语',
                '你在我身边，像是理所当然又像是奇迹',
                '有些距离刚刚好，不远不近',
            ],
            'anxiety': [
                '明天的事情明天再说',
                '悬而未决的事情最让人不安',
                '有些选择比没选择更困难',
                '等待的过程比结果更难熬',
            ],
        }
        texts = emotion_texts.get(emotion, RESERVED_EXPRESSIONS['modern_reserved'])
        return random.choice(texts)

    def _generate_vad(self, emotion: str) -> Tuple[float, float, float]:
        vad_map = {
            'sadness': (-0.65, -0.25, -0.45),
            'joy': (0.65, 0.35, 0.55),
            'loneliness': (-0.55, -0.15, -0.35),
            'love': (0.70, 0.30, 0.60),
            'anxiety': (-0.40, 0.45, -0.25),
            'confusion': (-0.15, 0.25, -0.15),
        }
        base = vad_map.get(emotion, (0, 0, 0))
        vad = tuple(np.clip(base[i] + np.random.randn() * 0.07, -1, 1) for i in range(3))
        return vad

    def generate_batch(self, num_samples: int) -> Dict:
        texts = []
        emotion_idxs = []
        vad_labels = []
        intensity_labels = []

        emotions = ['sadness', 'joy', 'loneliness', 'love', 'anxiety', 'confusion']

        for _ in range(num_samples):
            emotion = random.choice(emotions)
            sample = self.generate_single(emotion)
            texts.append(sample["text"])
            emotion_idxs.append(self.emotion_to_idx.get(emotion, 0))
            vad_labels.append(sample["vad"])
            intensity_labels.append(sample["intensity"])

        return {
            "texts": texts,
            "emotion_idxs": np.array(emotion_idxs),
            "vad_labels": np.array(vad_labels, dtype=np.float32),
            "intensity_labels": np.array(intensity_labels, dtype=np.float32),
        }


if __name__ == "__main__":
    print("=" * 50)
    print("Murakami & Reserved Style Generator Test")
    print("=" * 50)

    # 村上春树风格
    print("\n【村上春树风格】")
    murakami = MurakamiEmotionGenerator(seed=42)
    for emotion in ['joy', 'sadness', 'loneliness', 'love', 'anxiety']:
        sample = murakami.generate_single(emotion)
        print(f"{emotion}: {sample['text'][:40]}...")

    # 含蓄风格
    print("\n【含蓄表达风格】")
    reserved = ReservedEmotionGenerator(seed=42)
    for emotion in ['sadness', 'joy', 'loneliness', 'love']:
        sample = reserved.generate_single(emotion)
        print(f"{emotion}: {sample['text'][:40]}...")

    # 批量测试
    print("\n【批量生成测试】")
    data = murakami.generate_batch(500)
    print(f"村上风格: {len(data['texts'])} 样本")
    data2 = reserved.generate_batch(500)
    print(f"含蓄风格: {len(data2['texts'])} 样本")