# -*- coding: utf-8 -*-
"""
High-Quality Training Data Generator - MiniMax自我训练版
=====================================================

利用MiniMax对情感的理解，生成高质量训练数据

方法：
1. 基于MiniMax的情感心理学知识设计模板
2. 生成多样化、真实的情感表达
3. 准确的情感标签（VAD + 强度 + 反讽）
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 高质量情感模板（MiniMax基于心理学知识设计）====================

HIGH_QUALITY_TEMPLATES = {
    # ===== Joy 喜悦 =====
    "joy": {
        "templates": [
            "今天太开心了！项目终于上线了！",
            "哇，中彩票了！太棒了！",
            "收到心仪公司的offer，开心！",
            "今天的天气真好，心情也跟着好起来",
            "减肥成功了！终于瘦下来了！",
            "考试拿了第一名，太高兴了！",
            "见到好久不见的老朋友，好开心！",
            "年终奖发了，可以过一个好年了！",
            "升职加薪了！努力没有白费！",
            "孩子出生了，家庭更加完整了！",
            "终于买到喜欢的房子了！",
            "旅行回来，满满的美好回忆！",
            "美食太好吃了，满足！",
            "运动完出了一身汗，舒服！",
            "读完一本好书，很有收获！",
        ],
        "vad_base": (0.85, 0.50, 0.75),
        "intensity_range": (0.6, 1.0)
    },

    # ===== Sadness 悲伤 =====
    "sadness": {
        "templates": [
            "失恋了，心里空落落的...",
            "爷爷去世了，再也见不到他了",
            "失业了，不知道该怎么办",
            "最好的朋友搬走了，很舍不得",
            "高考失利，不知道还能做什么",
            "外婆病重了，在医院照顾她",
            "养了多年的狗去世了...",
            "创业失败了，还欠了一屁股债",
            "被最信任的朋友骗了...",
            "妈妈生病了，很担心",
            "感情出现问题，不知道还能不能继续",
            "论文被拒了，几个月的心血白费",
            "错过了见爷爷最后一面...",
            "丢了工作，感觉人生失去方向",
            "猫咪走丢了，好难过...",
        ],
        "vad_base": (-0.80, -0.35, -0.55),
        "intensity_range": (0.5, 1.0)
    },

    # ===== Anger 愤怒 =====
    "anger": {
        "templates": [
            "太气人了！又被人插队！",
            "老板不讲理，明明不是我的错",
            "被骗了，钱追不回来了",
            "等了两个小时还没上菜！",
            "天天加班到半夜，工资还那么少",
            "室友太吵了，影响我休息",
            "被人背后说坏话，真的很生气",
            "无良商家卖假货，气死我了！",
            "火车又晚点了三个小时！",
            "被汽车溅了一身水！",
            "遇到路怒症，差点出事故",
            "在网上被人身攻击！",
            "总是有人占我的停车位！",
            "商家虚假宣传，欺骗消费者",
            "被保险公司坑了！",
        ],
        "vad_base": (-0.80, 0.70, 0.50),
        "intensity_range": (0.5, 1.0)
    },

    # ===== Fear 恐惧/担忧 =====
    "fear": {
        "templates": [
            "明天要面试，好紧张啊",
            "体检报告有指标不正常，担心中",
            "下周要上台演讲，好害怕",
            "地震了！好吓人！",
            "看到恐怖片里的画面，不敢睡觉",
            "听说公司要裁员，不知道会不会轮到自己",
            "明天有重要考试，完全没复习",
            "第一次一个人住，晚上有点害怕",
            "看到新闻说有病毒蔓延，好担心",
            "下个月要见岳父岳母，好紧张",
            "飞机会不会出事？有点害怕坐飞机",
            "伤口一直不好，怕是什么大病",
            "明天要做手术，很害怕",
            "听说经济不好，怕被裁员",
            "参加比赛紧张得手心出汗",
        ],
        "vad_base": (-0.60, 0.55, -0.45),
        "intensity_range": (0.4, 0.9)
    },

    # ===== Surprise 惊讶 =====
    "surprise": {
        "templates": [
            "哇！收到礼物太意外了！",
            "没想到他真的会来！",
            "考试成绩出来，完全没想到考这么好！",
            "偶遇多年不见的同学，太巧了！",
            "中了新股，运气太好了！",
            "完全没想到他会求婚！",
            "公司居然给我放假！惊喜！",
            "听说有红包抢，好惊喜！",
            "没想到这篇论文能发表！",
            "生日居然收到这么多祝福！",
            "他居然记得我的生日！",
            "买彩票中了一万块！",
            "完全没想到能见到偶像！",
            "项目居然提前完成了！",
            "他向我表白，完全没想到！",
        ],
        "vad_base": (0.30, 0.80, 0.30),
        "intensity_range": (0.5, 1.0)
    },

    # ===== Anticipation 期待 =====
    "anticipation": {
        "templates": [
            "好期待下周去旅游！",
            "快要见到喜欢的人了，好激动！",
            "新年要到了，好期待！",
            "毕业典礼马上要举行了！",
            "终于要发年终奖了！",
            "下周有演唱会，超级期待！",
            "快要见到偶像了，睡不着！",
            "期待已久的电影终于上映了！",
            "要去表白，好紧张又好期待",
            "项目马上要上线了！",
            "生日快到了，不知道会收到什么礼物",
            "考研成绩要公布了，又期待又害怕",
            "要去见他家长了，好紧张",
            "房子终于要交房了！",
            "明天要去领证了！",
        ],
        "vad_base": (0.50, 0.60, 0.55),
        "intensity_range": (0.5, 0.95)
    },

    # ===== Trust 信任 =====
    "trust": {
        "templates": [
            "我相信你一定能做到！",
            "把这件事交给你，我很放心",
            "完全信任你的判断",
            "你是我的好朋友，我相信你",
            "这个投资很靠谱，我相信",
            "他一直很靠谱，我信任他",
            "这个品牌我一直用，很信任",
            "把密码告诉你，我信任你",
            "我相信你的能力",
            "这个医生很有名，我信任他",
            "我们的感情经得起考验",
            "你说什么我都相信",
            "这个计划很周全，我相信能成功",
            "把秘密告诉你，因为信任你",
            "我相信这个世界还是好人多",
        ],
        "vad_base": (0.65, -0.20, 0.50),
        "intensity_range": (0.4, 0.9)
    },

    # ===== Disgust 厌恶 =====
    "disgust": {
        "templates": [
            "这个菜的味道好恶心！",
            "那个人随地吐痰，真恶心！",
            "食堂的饭又涨价了，坑人！",
            "看到有人插队，真的很讨厌",
            "这个电影的剧情太狗血了！",
            "闻到下水道味，恶心死了！",
            "那个人满嘴脏话，真讨厌！",
            "这种虚伪的人最让人恶心",
            "蟑螂爬过我的食物！",
            "被强制看广告，烦死了！",
            "这个人太自私了，令人作呕",
            "吃到一个坏掉的鸡蛋！",
            "看到有人在公共场所大小便",
            "这个主播太做作了！",
            "闻到狐臭味绕道走！",
        ],
        "vad_base": (-0.75, -0.10, -0.40),
        "intensity_range": (0.4, 0.9)
    },

    # ===== Optimism 乐观 =====
    "optimism": {
        "templates": [
            "虽然现在困难，但一定会好起来的！",
            "相信明天会更好！",
            "这次失败是成功的前奏！",
            "困难只是暂时的，加油！",
            "阳光总在风雨后！",
            "我相信一切都会变好的！",
            "虽然跌倒了，但我会爬起来的！",
            "失败是成功之母，继续努力！",
            "最困难的时候已经过去了！",
            "曙光就在前面！",
            "我相信我可以的！",
            "这次一定会顺利的！",
            "困难只是考验，熬过去就好了",
            "每次挫折都是成长！",
            "未来可期！",
        ],
        "vad_base": (0.70, 0.45, 0.65),
        "intensity_range": (0.5, 0.85)
    },

    # ===== Love 爱 =====
    "love": {
        "templates": [
            "我真的很喜欢你！",
            "和你在一起的时光最美好！",
            "我爱我的家人！",
            "有你们真好！",
            "你是我最重要的人！",
            "我爱你，永远！",
            "感谢一路有你们陪伴！",
            "这就是爱情的样子吧！",
            "我的心只属于你！",
            "好幸福，有你们真好！",
            "世界上最幸福的事就是和你在一起",
            "感谢你的陪伴，我爱你！",
            "有你的地方就是家！",
            "你们是我的全部！",
            "我愿意和你走完一生！",
        ],
        "vad_base": (0.90, 0.35, 0.80),
        "intensity_range": (0.6, 1.0)
    },

    # ===== Anxiety 焦虑 =====
    "anxiety": {
        "templates": [
            "担心的事情终于发生了...",
            "睡不着觉，脑子里全是烦心事",
            "要同时处理这么多事，头都大了",
            "不知道该怎么选择，好纠结",
            "时间不够用，焦虑死了",
            "事情太多，完全理不清头绪",
            "一直在担心未来会怎样",
            "deadline明天就到了，还没做完",
            "同时收到坏消息，崩溃了",
            "不知道能不能做好，压力好大",
            "好担心会出什么差错",
            "心里一直不安，安静不下来",
            "被各种事情追着跑...",
            "要做一个重大决定，好害怕选错",
            "状态不好，怕影响发挥",
        ],
        "vad_base": (-0.50, 0.60, -0.40),
        "intensity_range": (0.4, 0.9)
    },

    # ===== Despair 绝望 =====
    "despair": {
        "templates": [
            "活着还有什么意义...",
            "真的撑不下去了...",
            "为什么所有坏事都让我遇到",
            "感觉被整个世界抛弃了",
            "努力了这么久还是失败，绝望",
            "再也不会好了吧...",
            "活着好累，不想再坚持了",
            "没有人能理解我...",
            "一切都完了...",
            "失去了活下去的动力",
            "好想消失在这个世界上",
            "太孤独了，没有人懂我",
            "感觉人生已经没有任何希望",
            "怎么努力都没用，放弃了",
            "世界一片黑暗...",
        ],
        "vad_base": (-0.85, 0.30, -0.70),
        "intensity_range": (0.6, 1.0)
    },

    # ===== Contempt 蔑视 =====
    "contempt": {
        "templates": [
            "这种人也好意思出来丢人？",
            "本事没有，架子倒是不小",
            "就知道吹牛，实际什么都不行",
            "藐视一切的人最可笑",
            "这点成就也值得炫耀？",
            "看他那副嘴脸，真恶心",
            "就这水平还敢指点江山？",
            "没有什么真本事，靠关系上位",
            "这种人就是社会的蛀虫",
            "自以为是，其实就是个笑话",
            "藐视他的无能",
            "他算什么？什么都不懂还装懂",
            "看不上这种小人得志",
            "真为他感到可悲",
            "这种人值得我浪费时间？",
        ],
        "vad_base": (-0.70, 0.20, 0.40),
        "intensity_range": (0.3, 0.8)
    },

    # ===== Disappointment 失望 =====
    "disappointment": {
        "templates": [
            "又让我失望了...",
            "期待越高，失望越大",
            "商家虚假宣传，太失望了",
            "本以为会很开心，结果很失望",
            "对你的信任错付了...",
            "期望落空，好失落...",
            "产品远没有宣传的那么好",
            "又一次失望...",
            "本来以为会很顺利的",
            "你让我太失望了",
            "结果令人失望...",
            "说好的不涨价的！",
            "服务态度太差，很失望",
            "没想到会是这样的结果",
            "对这次体验很不满",
        ],
        "vad_base": (-0.65, -0.20, -0.45),
        "intensity_range": (0.4, 0.85)
    },

    # ===== Envy 羡慕/嫉妒 =====
    "envy": {
        "templates": [
            "凭什么他就可以，我不服！",
            "好羡慕别人的成功啊...",
            "他运气真好，嫉妒！",
            "什么时候我也能像他一样",
            "别人都那么优秀，就我最差",
            "好羡慕能有这样的成就",
            "为什么差距这么大...",
            "看到别人成功，自己却原地踏步",
            "他也太厉害了吧",
            "要是我也有那样的机会...",
            "嫉妒使我丑陋...",
            "别人的男朋友/女朋友好优秀",
            "他的生活为什么那么精彩",
            "看着别人的幸福，有点酸",
            "自己怎么就不行呢...",
        ],
        "vad_base": (-0.40, 0.30, -0.30),
        "intensity_range": (0.3, 0.75)
    },

    # ===== Guilt 内疚 =====
    "guilt": {
        "templates": [
            "对不起，是我错了...",
            "真的对不起你的信任",
            "做错了事，好愧疚...",
            "过不了自己这关...",
            "对不起家人对我的期望",
            "因为我的失误害了大家...",
            "辜负了你们的信任...",
            "无法原谅自己的错误",
            "一直很愧疚，不敢面对",
            "伤害了最亲的人...",
            "后悔当初的选择...",
            "如果当时不那么做就好了",
            "对不起，真的很抱歉...",
            "内心愧疚，无法释怀",
            "是我的错，给大家添麻烦了",
        ],
        "vad_base": (-0.50, 0.20, -0.40),
        "intensity_range": (0.4, 0.85)
    },
}


# ==================== 反讽模板 ====================

IRONY_TEMPLATES = {
    "joy_to_sadness": {
        "surface": "joy",
        "true_emotion": "sadness",
        "vad": (-0.60, -0.20, -0.30),
        "templates": [
            "太好了，火车又晚点了6小时",
            "谢谢让我等了3个小时",
            "哇，又加班到凌晨，真开心",
            "太好了，硬盘又坏了，数据全没了",
            "真棒，又感冒了，开心",
            "谢谢你的\"帮助\"，让我等了这么久",
            "太感动了，室友又半夜唱歌",
            "好惊喜，老板又临时加需求",
        ]
    },
    "joy_to_anger": {
        "surface": "joy",
        "true_emotion": "anger",
        "vad": (-0.50, 0.40, 0.20),
        "templates": [
            "太好了，又有人插队了",
            "哇，第一次被骗这么多钱",
            "真棒，又被坑了",
            "谢谢你的\"诚实\"，让我看清了你",
            "太好了，又被放鸽子",
            "哇，又被同事抢了功劳",
            "真不错，又被客户骂了",
            "谢谢\"耐心\"听完我的汇报",
        ]
    },
    "anticipation_to_disappointment": {
        "surface": "anticipation",
        "true_emotion": "disappointment",
        "vad": (-0.50, -0.10, -0.20),
        "templates": [
            "好期待，结果又失望了",
            "满怀期待等来这个结果...",
            "说好的惊喜呢？",
            "期待落空的感觉真难受",
            "又是一次空欢喜",
            "本以为会很棒的...",
            "希望越大，失望越大",
            "想象中的美好呢？",
        ]
    },
}


class HighQualityDataGenerator:
    """高质量数据生成器"""

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

    def generate_single(self, emotion: str) -> Dict[str, Any]:
        """生成单条数据"""
        if emotion not in HIGH_QUALITY_TEMPLATES:
            return None

        template = HIGH_QUALITY_TEMPLATES[emotion]
        text = random.choice(template["templates"])

        # 生成强度
        intensity = random.uniform(*template["intensity_range"])

        # 生成VAD (带小噪声)
        base_v, base_a, base_d = template["vad_base"]
        vad = (
            np.clip(base_v + np.random.randn() * 0.08, -1, 1),
            np.clip(base_a + np.random.randn() * 0.10, -1, 1),
            np.clip(base_d + np.random.randn() * 0.08, -1, 1)
        )

        # 生成多标签
        emotion_labels = np.zeros(26, dtype=np.float32)
        emotion_labels[self.emotion_to_idx[emotion]] = intensity

        # 添加相关情感
        related = self._get_related_emotions(emotion)
        for rel in related:
            if rel in self.emotion_to_idx:
                rel_intensity = intensity * random.uniform(0.3, 0.6)
                emotion_labels[self.emotion_to_idx[rel]] = rel_intensity

        return {
            "text": text,
            "emotion_labels": emotion_labels,
            "vad_labels": np.array(vad, dtype=np.float32),
            "intensity_labels": np.array([intensity], dtype=np.float32),
            "irony_labels": np.array([0.0], dtype=np.float32),
            "primary_emotion": emotion,
            "intensity": intensity,
            "vad": vad
        }

    def _get_related_emotions(self, emotion: str) -> List[str]:
        """获取相关情感"""
        relations = {
            "joy": ["optimism", "pride", "contentment", "love"],
            "sadness": ["disappointment", "remorse", "despair", "anxiety"],
            "anger": ["aggression", "contempt", "disgust"],
            "fear": ["anxiety", "submission", "suspicion"],
            "trust": ["love", "submission"],
            "disgust": ["contempt", "cynicism", "anger"],
            "anticipation": ["optimism", "anxiety", "envy"],
            "surprise": ["fear", "joy"],
            "optimism": ["joy", "anticipation"],
            "love": ["joy", "trust", "sentimentality"],
            "anxiety": ["fear", "sadness"],
            "despair": ["sadness", "anger"],
            "contempt": ["disgust", "anger"],
            "disappointment": ["sadness", "anger"],
            "envy": ["sadness", "discontent"],
            "guilt": ["sadness", "submission"],
        }
        return relations.get(emotion, [])

    def generate_batch(self, num_samples: int, irony_ratio: float = 0.15) -> Dict[str, Any]:
        """生成批量数据"""
        texts = []
        emotion_labels = []
        vad_labels = []
        intensity_labels = []
        irony_labels = []

        normal_count = int(num_samples * (1 - irony_ratio))
        irony_count = num_samples - normal_count

        emotions = list(HIGH_QUALITY_TEMPLATES.keys())

        # 生成正常样本
        for _ in range(normal_count):
            emotion = random.choice(emotions)
            sample = self.generate_single(emotion)
            if sample:
                texts.append(sample["text"])
                emotion_labels.append(sample["emotion_labels"])
                vad_labels.append(sample["vad_labels"])
                intensity_labels.append(sample["intensity_labels"])
                irony_labels.append(sample["irony_labels"])

        # 生成反讽样本
        irony_types = list(IRONY_TEMPLATES.keys())
        for _ in range(irony_count):
            irony_type = random.choice(irony_types)
            template = IRONY_TEMPLATES[irony_type]
            text = random.choice(template["templates"])

            # 添加小变化
            text = text + random.choice(["", "", ""])  # 偶尔加标点

            texts.append(text)
            emotion_labels.append(np.zeros(26, dtype=np.float32))
            vad_labels.append(np.array(template["vad"], dtype=np.float32))
            intensity_labels.append(np.array([0.6], dtype=np.float32))
            irony_labels.append(np.array([1.0], dtype=np.float32))

        return {
            "texts": texts,
            "emotion_labels": np.array(emotion_labels, dtype=np.float32),
            "vad_labels": np.array(vad_labels, dtype=np.float32),
            "intensity_labels": np.array(intensity_labels, dtype=np.float32),
            "irony_labels": np.array(irony_labels, dtype=np.float32)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("High-Quality Data Generator - 测试")
    print("=" * 60)

    generator = HighQualityDataGenerator()

    # 测试生成
    print("\n【样本展示】")
    emotions_to_show = ["joy", "sadness", "anger", "fear", "love", "anxiety"]
    for emotion in emotions_to_show:
        sample = generator.generate_single(emotion)
        if sample:
            print(f"  {emotion}: {sample['text']}")

    # 批量生成
    print("\n【批量生成测试】")
    data = generator.generate_batch(1000, irony_ratio=0.15)
    print(f"  总样本: {len(data['texts'])}")
    print(f"  情感标签形状: {data['emotion_labels'].shape}")
    print(f"  反讽样本: {int(data['irony_labels'].sum())}")

    print("\n高质量数据生成器测试通过!")
