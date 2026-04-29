# -*- coding: utf-8 -*-
"""
Literary-Style Emotional Text Generator
基于红楼梦等经典文学作品的情感表达风格
"""

import random
import numpy as np
from typing import Dict, List, Tuple

# 红楼梦风格的情感文本库
LITERARY_TEMPLATES = {
    'joy': [
        '黛玉拈着花枝，心中欢喜，嘴角不禁微微扬起',
        '宝钗见姐妹们来了，心中自是欢喜不尽',
        '贾母听了这话，心中大喜，忙命人摆宴庆贺',
        '黛玉听宝玉说了这番话，心中又喜又羞',
        '探春笑道：竟是件大喜事！众人皆欢喜起来',
        '凤姐儿听说老太太高兴，自己也觉得欢喜',
        '宝玉见林妹妹笑了，心中欢喜，便说道',
        '姐妹们一处吟诗作对，倒也欢喜',
        '老太太听了这个消息，喜出望外',
        '贾政见儿子争气，心中暗喜',
        '黛玉心下暗喜道：难得遇见知音',
        '宝琴笑着拍手，众人看着都欢喜',
    ],
    'sadness': [
        '黛玉葬花，泪洒花锄，呜呜咽咽哭了起来',
        '宝玉听了这话，心中凄凉，不觉滴下泪来',
        '黛玉听罢，泪流满面，哽咽得说不出话来',
        '宝钗虽劝慰，黛玉却越发伤心，哭得肝肠寸断',
        '凤姐儿听闻噩耗，泪如雨下，痛不欲生',
        '贾母听闻此言，哭得死去活来',
        '黛玉想起自己的身世，不免又伤心起来',
        '宝玉看着林妹妹这般模样，心中甚是难过',
        '探春守在窗前，暗自垂泪',
        '黛玉的病一日重似一日，众人看了都暗自伤心',
        '贾政叹了口气，不觉落下泪来',
        '黛玉冷笑道：我这里早就没什么可留恋的了说罢又落下泪来',
    ],
    'anger': [
        '黛玉听了，气得浑身发抖，冷笑道',
        '凤姐儿大怒道：竟有这等事！',
        '贾政听了，气得脸色发青，怒道',
        '黛玉心中气恼，却又不好发作，只冷笑了一声',
        '宝玉见林妹妹动了气，心中惶恐，连忙赔罪',
        '贾母大怒道：这等下作的东西，怎能进我的门！',
        '黛玉听说，越发恼了，便道',
        '凤姐儿气得摔了茶杯，众人都不敢言语',
        '宝玉见父亲发怒，慌忙跪下',
        '黛玉冷笑道：横竖如今也没人真心待我',
        '贾政拿起棍子就要打，宝玉慌忙躲避',
        '凤姐儿怒道：都是些没良心的东西！',
    ],
    'fear': [
        '黛玉听说，大惊失色，忙问端的',
        '凤姐儿听了这话，吓得脸色惨白',
        '宝玉拉住林妹妹的衣袖道：林妹妹别怕，有我在',
        '黛玉心中害怕，却又不好意思说出来',
        '贾母听闻此言，吓得浑身发抖，连话都说不出来',
        '凤姐儿吓得不敢吱声，生怕惹出祸端',
        '黛玉看了那信，吓得魂飞魄散',
        '宝玉见林妹妹脸色不对，忙问怎么了',
        '贾政听说此事，吓得跌足长叹',
        '黛玉听说，吓得连忙把信藏了',
        '众人听了，都吓得不敢说话',
        '凤姐儿心下惊恐，强作镇定',
    ],
    'love': [
        '宝玉看着林妹妹，心中涌起无限柔情',
        '黛玉见宝玉待她这般好，心中又喜又羞',
        '宝黛二人情投意合，心意相通',
        '黛玉低头不语，心中暗喜，情意绵绵',
        '宝玉轻声道：林妹妹，你放心',
        '黛玉听闻此言，心中一暖，落下泪来',
        '二人互诉衷肠，情意更深',
        '宝玉握着林妹妹的手，柔情万千',
        '黛玉心下暗想：他若真待我好，我便...',
        '宝黛之间的情意，虽未明说，却心照不宣',
        '凤姐儿打趣道：宝兄弟心里只有林妹妹',
        '黛玉听了，羞红了脸，心中却甚是欢喜',
    ],
    'anxiety': [
        '黛玉这几日茶饭不思，心中焦虑不安',
        '宝玉见林妹妹脸色不好，心中忧虑',
        '凤姐儿心下焦急，却又不好说出来',
        '黛玉想起自己的病势，心中焦虑',
        '贾母见宝玉这般，心中忧虑重重',
        '黛玉担心自己的病好不了，越发焦虑',
        '凤姐儿为府里的事日夜操心，心下焦虑',
        '宝玉担心林妹妹的病，整日忧心忡忡',
        '黛玉心烦意乱，不知如何是好',
        '众人见黛玉这般，都心下担忧',
        '贾政为儿子的前程忧虑不已',
        '黛玉夜里睡不着，心中烦躁不安',
    ],
    'despair': [
        '黛玉冷笑道：我如今是没人疼的了说罢泪如雨下',
        '凤姐儿心想，如今这局面，已是无法挽回了',
        '黛玉绝望道：与其这样，不如死了干净',
        '宝玉听说，心中绝望，不觉大哭起来',
        '黛玉跪在窗前，泪流满面，心如死灰',
        '凤姐儿看着这般光景，心下绝望',
        '黛玉的病一日重似一日，众人都有了不好的预感',
        '宝玉见林妹妹这般模样，心中悲痛欲绝',
        '黛玉凄然道：我知道我的命不久矣',
        '众人看了这情形，心下都知道不好了',
        '贾母哭道：这可怎么是好，老天爷要收人呐！',
        '黛玉叹了口气，绝望地闭上了眼睛',
    ],
    'disgust': [
        '黛玉见了他那副嘴脸，心中厌烦',
        '凤姐儿冷笑道：真真让人恶心',
        '黛玉不屑道：这等小人，我也懒得计较',
        '宝玉听了，心中甚是厌恶',
        '黛玉冷眼旁观，心中甚是鄙夷',
        '凤姐儿啐了一口：什么东西！',
        '黛玉见他如此虚伪，心中不屑',
        '宝玉心想，这等人实在可恶',
        '黛玉蹙眉道：一股子俗气',
        '凤姐儿撇撇嘴：我正眼都不想瞧他',
        '黛玉冷笑道：装什么正经',
        '宝玉心下厌恶，却不好发作',
    ],
    'contempt': [
        '黛玉冷笑道：他算什么东西，也配和我说话？',
        '凤姐儿轻蔑道：我当他多厉害，不过如此',
        '黛玉嗤之以鼻，不屑一顾',
        '宝玉心下暗道：这人真是可笑',
        '黛玉斜眼看着那人，满是轻蔑',
        '凤姐儿道：那人就是个没本事的',
        '黛玉冷声道：不过是个攀附权贵的',
        '宝玉心道：藐视天下英雄',
        '黛玉轻哼一声：不值一提',
        '凤姐儿撇着嘴，满是鄙视',
    ],
    'surprise': [
        '黛玉大惊道：竟有这等事？',
        '凤姐儿吃惊道：这可是没想到的',
        '宝玉惊讶道：林妹妹怎么知道的？',
        '黛玉惊讶道：竟有这般巧的事',
        '众人惊讶不已，都说没想到',
        '宝玉惊呼：这也太神了！',
        '黛玉惊道：怎么会有这种事！',
        '凤姐儿惊得目瞪口呆',
        '宝玉惊问道：当真如此？',
        '黛玉惊道：我竟不知道有这事！',
        '众人惊讶地看着，不知说什么好',
        '凤姐儿惊道：我的天哪！',
    ],
    'anticipation': [
        '黛玉心中暗暗期待着那一天',
        '宝玉盼望着和林妹妹再见一面',
        '凤姐儿等着看好戏',
        '黛玉期待着宝玉的心意',
        '宝玉盼着老太太能准了这门亲事',
        '黛玉心中暗喜，盼望的日子终于要到了',
        '凤姐儿等着瞧他们的下场',
        '宝玉期待着园中的诗会',
        '黛玉盼着那人早日归来',
        '凤姐儿等着看热闹',
        '众人期待着中秋佳节的到来',
        '宝玉盼着能和林妹妹一处读书',
    ],
    'trust': [
        '黛玉心中暗想，宝玉是个靠得住的',
        '凤姐儿心想，这事交给他办定是妥当的',
        '宝玉对林妹妹道：你放心，我定不负你',
        '黛玉见宝玉这般说，心下稍安',
        '凤姐儿道：这这事交给他，我放心',
        '宝玉道：林妹妹，你信我',
        '黛玉心中暗暗点头，知道他是可信的',
        '凤姐儿对平儿道：他是信得过的人',
        '黛玉见他说得真诚，便信了',
        '宝玉心道：林妹妹总是信我的',
    ],
}


class LiteraryDataGenerator:
    """文学风格情感数据生成器"""

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

    def generate_sample(self, emotion: str = None) -> Dict:
        """生成单条样本"""
        if emotion is None:
            emotion = random.choice(list(LITERARY_TEMPLATES.keys()))

        if emotion not in LITERARY_TEMPLATES:
            return None

        templates = LITERARY_TEMPLATES[emotion]
        text = random.choice(templates)

        # 添加变体
        if random.random() < 0.3:
            suffix = random.choice([
                '，不禁',
                '，于是',
                '，心中想着',
                '，不觉',
                '，却又',
                '，只是',
            ])
            extra = random.choice([
                '叹了口气',
                '落下泪来',
                '微微一笑',
                '默然不语',
                '暗自垂泪',
                '冷笑一声',
                '喜极而泣',
                '心下暗想',
            ])
            text = text + suffix + extra

        # 生成VAD标签
        vad_map = {
            'joy': (0.8, 0.5, 0.7),
            'sadness': (-0.8, -0.3, -0.5),
            'anger': (-0.8, 0.7, 0.5),
            'fear': (-0.6, 0.6, -0.4),
            'love': (0.9, 0.4, 0.8),
            'anxiety': (-0.5, 0.6, -0.4),
            'despair': (-0.9, 0.3, -0.7),
            'disgust': (-0.7, -0.2, -0.4),
            'contempt': (-0.6, 0.2, 0.4),
            'surprise': (0.3, 0.8, 0.3),
            'anticipation': (0.5, 0.6, 0.5),
            'trust': (0.6, -0.2, 0.5),
        }

        base_vad = vad_map.get(emotion, (0, 0, 0))
        vad = tuple(np.clip(v + np.random.randn() * 0.1, -1, 1) for v in base_vad)
        intensity = random.uniform(0.5, 0.9)

        return {
            "text": text,
            "emotion": emotion,
            "vad": vad,
            "intensity": intensity,
        }

    def generate_batch(self, num_samples: int, emotions: List[str] = None) -> Dict:
        """生成批量样本"""
        if emotions is None:
            emotions = list(LITERARY_TEMPLATES.keys())

        texts = []
        emotion_idxs = []
        vad_labels = []
        intensity_labels = []

        for _ in range(num_samples):
            emotion = random.choice(emotions)
            sample = self.generate_sample(emotion)
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
    gen = LiteraryDataGenerator()
    print("红楼梦风格情感数据生成器测试")
    print("=" * 50)

    # 测试生成
    emotions_to_show = ['joy', 'sadness', 'anger', 'love', 'anxiety', 'despair']
    for emotion in emotions_to_show:
        sample = gen.generate_sample(emotion)
        if sample:
            print(f"{emotion}: {sample['text']}")

    # 批量生成
    print("\n批量生成测试:")
    data = gen.generate_batch(1000)
    print(f"总样本: {len(data['texts'])}")
    print(f"标签形状: {data['emotion_idxs'].shape}")