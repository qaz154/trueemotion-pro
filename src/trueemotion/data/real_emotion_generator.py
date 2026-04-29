# -*- coding: utf-8 -*-
"""
Realistic Chinese Emotional Text Generator v2
基于真实中文互联网表达习惯的情感文本生成
"""

import random
import numpy as np
from typing import Dict, List, Tuple

# ==================== 真实情感表达词库 ====================

# 网络用语和口语化表达
INTERNET_EMOJI_MAP = {
    'joy': ['哈哈哈', '笑死', '太逗了', '666', '哈哈', '开心', '美滋滋', '爽', '耶', '噢耶', '开心到飞起'],
    'sadness': ['呜呜', '扎心', '难过', '失落', '蓝瘦', '香菇', '想哭', '泪目', '心塞', '郁闷', '丧'],
    'anger': ['气死', '怒', '草', '靠', '尼玛', '无语', '服了', '呵呵', '滚', '呸'],
    'fear': ['怕怕', '慌', '瑟瑟发抖', '紧张', '虚', '怂', '怕', '不敢'],
    'surprise': ['卧槽', '牛', '厉害', '震惊', '天哪', '不敢相信', '惊了', '真的假的'],
    'anxiety': ['焦虑', '慌', '急', '烦', '愁', '秃', '秃头', '失眠', '压力山大'],
    'love': ['甜', '齁', '酸', '慕', 'Kiss', '么么哒', 'love', '喜欢', '心动', '上头'],
    'disgust': ['呕', '恶心', '吐', '嫌弃', '辣眼睛', '尴尬', '无语', '假'],
    'despair': ['绝望', '废了', '完了', '没救了', '自闭', '佛系', '躺平', '摆烂'],
    'contempt': ['呵呵', '就这', '就这?', '菜', '辣鸡', '垃圾', '渣渣', '废物'],
    'envy': ['酸', '慕', '嫉妒', '柠檬精', '馋', '酸了', '眼红'],
    'guilt': ['愧疚', '自责', '对不起', '抱歉', '过意不去', '不好意思'],
    'trust': ['信', '靠谱', '稳', '放心', '交给我', '靠谱的'],
    'anticipation': ['期待', '搓手', '等不及', '渴望', '想要', '希望'],
}

# 真实生活场景情感表达
SCENE_TEMPLATES = {
    'work': {
        'joy': [
            '今天项目终于上线了，太开心了！',
            '老板居然给我升职了！意想不到！',
            '发工资了，可以好好犒劳一下自己',
            '工作终于得到了认可，开心！',
            '今天的presentation很成功，领导夸我了',
            '完成了一个大单，提成应该不少',
            '终于搞定这个难缠的客户了',
            '团队的配合越来越默契了，工作很开心',
        ],
        'sadness': [
            '加班到凌晨三点，明天还要早起，心累',
            '项目又被砍了，几个月的努力白费了',
            '被领导当众批评了，好丢脸',
            '工作压力太大，喘不过气来',
            '又失业了，不知道还能做什么',
            '和同事闹矛盾了，氛围很尴尬',
            '甲方爸爸又改需求了，想哭',
            '试用期没过，又要开始找工作了',
        ],
        'anger': [
            '甲方又改需求了！气死了！',
            '同事甩锅给我，真的很无语',
            '老板画的饼永远吃不到',
            '天天加班还没加班费，黑心公司',
            '同事在背后说我坏话，被我听到了',
            '面试被放鸽子，等了两小时人没来',
            '公司又裁员了，惶惶不可终日',
        ],
        'anxiety': [
            '明天还有重要汇报，睡不着怎么办',
            '月底KPI完不成，要被炒了',
            '简历投了几十份，一个面试都没有',
            '试用期快到了，不知道能不能转正',
            '工作越来越多，一个人根本做不完',
            '害怕接到来历不明的电话',
            '不知道自己还能在这个行业干多久',
        ],
    },
    'relationship': {
        'joy': [
            '男/女朋友今天给我买了礼物，好感动',
            '和喜欢的人聊天聊到半夜都不想睡',
            '约会很成功，她/他好像对我也有意思',
            '异地恋终于结束了，可以天天在一起了',
            '收到分手后的第一封情书',
            '他/她终于主动约我了！',
            '和男/女朋友吵架后和好了，更珍惜彼此了',
        ],
        'sadness': [
            '分手一个月了还是放不下',
            '暗恋的人脱单了，心碎',
            '和男/女朋友吵架了，不知道怎么办',
            '他说我们不合适，想哭',
            '异地的第365天，想他/她了',
            '家人安排的相亲又失败了',
            '被发好人卡，说只想做朋友',
        ],
        'anger': [
            '男/女朋友已读不回，气死我了',
            '被绿了，没想到他/她是这种人',
            '他/她居然瞒着我见前男/女朋友',
            '说好一起过节，又被放鸽子了',
            '他/她跟别人暧昧被我抓到了',
            '每次吵架都是我先道歉，累了',
        ],
        'anxiety': [
            '不知道他/她是不是真的喜欢我',
            '我们在一起一年了，要不要结婚',
            '感觉他/她最近对我冷淡了',
            '我不知道该不该表白',
            '他说需要空间，我是不是被分手了',
            '不知道他/她喜欢什么样的',
        ],
    },
    'life': {
        'joy': [
            '今天天气真好，心情也跟着好起来',
            '终于瘦了五斤，开心！',
            '抢到了演唱会门票，激动！',
            '周末要去旅行了，期待！',
            '中了新股，运气太好了！',
            '买到了想要的手机，好开心',
            '今天遇到好人帮我捡了钥匙',
            '养的植物开花了，好有成就感',
        ],
        'sadness': [
            '感冒了一周还没好，人都虚了',
            '丢了我最喜欢的那把伞',
            '下雨天摔了一跤，好痛',
            '一个人过生日，有点凄凉',
            '家里下水道堵了，好烦',
            '快递丢了，还不赔钱',
            '长胖了五斤，想哭',
        ],
        'anger': [
            '隔壁装修一大早就开始吵',
            '外卖又送错了，还不承认',
            '共享单车是坏的，还扣我钱',
            '电梯坏了，爬了二十楼',
            '邻居半夜唱歌，吵死了',
            '出租车司机绕路多收我钱',
            '理发店给我剪毁了，想哭',
        ],
        'anxiety': [
            '体检报告有几个指标不正常',
            '下个月房租又要涨了',
            '银行卡余额不足了',
            '信用卡还款日到了',
            '不知道明天会不会下雨',
            '预约的医生会不会又取消',
        ],
    },
    'study': {
        'joy': [
            '考试终于结束了，可以放松了',
            '拿到心仪大学的offer了！',
            '论文被导师夸了，有被鼓励到',
            '考研成绩过了！激动！',
            '拿到了奖学金，好开心',
            '参加比赛拿了一等奖',
            '终于考过了驾照，好开心',
        ],
        'sadness': [
            '考研失败了，不知道该二战还是工作',
            '论文又被导师打回来了',
            '期末考试没考好',
            '英语六级又没过',
            '上课迟到被点名了',
            '做毕业设计完全没头绪',
        ],
        'anxiety': [
            '马上就高考了，好紧张',
            '还有一周就交论文了',
            '申请留学的截止日期快到了',
            '不知道能不能考上研究生',
            '期末考试周要到了',
            '答辩会不会被导师刁难',
        ],
    },
    'social': {
        'joy': [
            '和好久不见的朋友聚会，好开心',
            '被人夸今天的穿搭好看',
            '发朋友圈收到很多赞',
            '在群里发红包，手气王是我',
            '帮到别人了，很开心',
            '和老朋友聊天聊到忘记时间',
        ],
        'sadness': [
            '发消息没人回，是不是被讨厌了',
            '朋友圈没人点赞，是不是太无聊了',
            '群里聊天我说话没人接',
            '想找个人吃饭都找不到',
            '感觉自己越来越孤僻了',
        ],
        'anxiety': [
            '等下要上台演讲，好紧张',
            '要不要给领导送礼，好纠结',
            '群里没人理我，是不是我说错话了',
            '要不要参加同事聚会，怕尴尬',
        ],
    },
}


class RealEmotionGenerator:
    """真实情感文本生成器"""

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

        # 扩展情感词库
        self.internet_expressions = INTERNET_EMOJI_MAP
        self.scenes = SCENE_TEMPLATES

    def _generate_variation(self, text: str, emotion: str) -> str:
        """生成文本变体"""
        # 随机添加前缀或后缀
        if random.random() < 0.4:
            prefixes = [
                '真的', '其实', '说实话', '没想到', '居然', '竟然',
                '今天', '现在', '唉', '哎', '啊', '噢', '我去',
            ]
            suffixes = [
                '了', '啊', '呀', '呢', '吧', '...', '。',
                '好么', '怎么办', '...', '呜呜', '哈哈哈',
            ]
            if random.random() < 0.5:
                text = random.choice(prefixes) + text
            else:
                text = text + random.choice(suffixes)

        # 添加情感强调词
        if random.random() < 0.3:
            emotions_map = {
                'joy': ['超', '巨', '太', '真', '好'],
                'sadness': ['好', '特别', '十分', '非常', '太'],
                'anger': ['超', '巨', '真', '太', '极其'],
                'anxiety': ['好', '特别', '有点', '有些', '越来越'],
                'fear': ['有点', '有些', '好', '特别', '非常'],
                'love': ['超', '巨', '好', '太', '真'],
            }
            words = emotions_map.get(emotion, ['很'])
            if text.startswith(('超', '巨', '太', '真', '好', '特别')):
                pass
            else:
                text = random.choice(words) + text

        # 添加感叹号/问号
        if random.random() < 0.3:
            if '？' not in text and '!' not in text:
                text = text + random.choice(['!', '!', '...', '。'])

        return text

    def _generate_sarcasm(self, emotion: str) -> str:
        """生成反讽文本"""
        sarcasm_map = {
            'joy': [
                '太好了，火车又晚点了6小时',
                '谢谢，让我等了3个小时',
                '太开心了，硬盘又坏了',
                '哇，又加班到凌晨，真爽',
            ],
            'sadness': [
                '好开心，又被放鸽子了',
                '太好了，又被人踩了一脚',
                '谢谢你的\"帮助\"，让我等了这么久',
            ],
            'anger': [
                '太好了，又有人插队了',
                '哇，第一次被骗了这么多钱',
                '真棒，又被坑了',
                '谢谢\"耐心\"听完我的汇报',
            ],
            'anticipation': [
                '好期待，结果又落空了',
                '满怀期待等来这个结果',
                '说好的惊喜呢？',
                '希望越大失望越大',
            ],
        }
        templates = sarcasm_map.get(emotion, [])
        if templates and random.random() < 0.3:
            return random.choice(templates)
        return None

    def _generate_vad(self, emotion: str) -> Tuple[float, float, float]:
        """生成VAD标签"""
        vad_map = {
            'joy': (0.85, 0.50, 0.75),
            'sadness': (-0.80, -0.35, -0.55),
            'anger': (-0.80, 0.70, 0.50),
            'fear': (-0.60, 0.55, -0.45),
            'surprise': (0.30, 0.80, 0.30),
            'anticipation': (0.50, 0.60, 0.55),
            'trust': (0.65, -0.20, 0.50),
            'disgust': (-0.75, -0.10, -0.40),
            'optimism': (0.70, 0.45, 0.65),
            'love': (0.90, 0.35, 0.80),
            'guilt': (-0.50, 0.20, -0.40),
            'anxiety': (-0.50, 0.60, -0.40),
            'despair': (-0.85, 0.30, -0.70),
            'contempt': (-0.70, 0.20, 0.40),
            'disappointment': (-0.65, -0.20, -0.45),
            'envy': (-0.40, 0.30, -0.30),
        }
        base = vad_map.get(emotion, (0, 0, 0))
        # 添加随机噪声
        vad = tuple(np.clip(base[i] + np.random.randn() * 0.1, -1, 1) for i in range(3))
        return vad

    def generate_single(self, emotion: str = None) -> Dict:
        """生成单条样本"""
        if emotion is None:
            emotion = random.choice(list(self.internet_expressions.keys()))

        # 尝试生成反讽
        if random.random() < 0.15:
            sarcasm = self._generate_sarcasm(emotion)
            if sarcasm:
                return {
                    "text": sarcasm,
                    "emotion": emotion,
                    "vad": self._generate_vad(emotion),
                    "intensity": random.uniform(0.5, 0.8),
                    "is_sarcasm": True
                }

        # 从场景模板生成
        if random.random() < 0.5:
            scene_name = random.choice(list(self.scenes.keys()))
            scene = self.scenes[scene_name]
            if emotion in scene:
                templates = scene[emotion]
                text = random.choice(templates)
                text = self._generate_variation(text, emotion)
                return {
                    "text": text,
                    "emotion": emotion,
                    "vad": self._generate_vad(emotion),
                    "intensity": random.uniform(0.4, 0.9),
                    "is_sarcasm": False
                }

        # 从网络用语生成
        if emotion in self.internet_expressions:
            expr = random.choice(self.internet_expressions[emotion])
            # 生成更长的表达
            prefixes = [
                '', '', '',
                '今天', '现在', '唉', '哎',
                '真的是', '简直', '太tm', '我去',
            ]
            suffixes = [
                '', '', '',
                '了', '啊', '呀', '...', '。', '的状态',
            ]
            text = random.choice(prefixes) + expr + random.choice(suffixes)
            if random.random() < 0.3:
                text = text + '，' + random.choice([
                    '好烦', '开心', '难过', '郁闷', '纠结',
                    '怎么办', '求解', '呜呜', '哈哈哈',
                ])
            return {
                "text": text,
                "emotion": emotion,
                "vad": self._generate_vad(emotion),
                "intensity": random.uniform(0.4, 0.9),
                "is_sarcasm": False
            }

        return None

    def generate_batch(self, num_samples: int) -> Dict:
        """生成批量样本"""
        texts = []
        emotion_idxs = []
        vad_labels = []
        intensity_labels = []

        emotions = list(set(
            list(self.internet_expressions.keys()) +
            [e for scenes in self.scenes.values() for e in scenes.keys()]
        ))

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


if __name__ == "__main__":
    gen = RealEmotionGenerator()
    print("=" * 50)
    print("RealEmotion Generator v2 Test")
    print("=" * 50)

    # 展示各情感样本
    emotions = ['joy', 'sadness', 'anger', 'anxiety', 'love', 'fear']
    for emotion in emotions:
        sample = gen.generate_sample(emotion)
        if sample:
            print(f"{emotion}: {sample['text']}")

    # 测试批量生成
    print("\n批量生成:")
    data = gen.generate_batch(1000)
    print(f"生成 {len(data['texts'])} 样本")