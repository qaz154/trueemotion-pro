# -*- coding: utf-8 -*-
"""
Multi-Label Emotion Analyzer
多标签情感分析器 - 支持输出多个情感标签
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class MultiLabelEmotionResult:
    """多标签情感结果"""
    primary_emotion: str
    all_emotions: Dict[str, float]  # {emotion: confidence}
    top_k: List[Tuple[str, float]]  # [(emotion, confidence), ...]
    vad: Tuple[float, float, float]
    confidence: float
    source: str
    is_multi_label: bool  # 是否检测到多个情感

    def get_primary_emotion(self) -> str:
        return self.primary_emotion

    def get_top_emotions(self, k: int = 3) -> List[Tuple[str, float]]:
        return self.top_k[:k]

    def has_emotion(self, emotion: str, threshold: float = 0.3) -> bool:
        return self.all_emotions.get(emotion, 0.0) >= threshold

    def to_dict(self) -> Dict:
        return {
            'primary_emotion': self.primary_emotion,
            'all_emotions': self.all_emotions,
            'top_emotions': self.top_k[:5],
            'vad': self.vad,
            'confidence': self.confidence,
            'source': self.source,
            'is_multi_label': self.is_multi_label
        }


class MultiLabelEmotionDetector:
    """
    多标签情感检测器

    规则系统 + 关键词匹配，支持多标签输出
    """

    # 情感关键词（带强度权重）
    EMOTION_KEYWORDS = {
        'joy': {
            '开心': 0.9, '高兴': 0.9, '快乐': 0.9, '欢乐': 0.8, '喜悦': 0.9,
            '愉快': 0.8, '爽': 0.8, '耶': 0.7, '哈哈': 0.8, '哈哈哈': 0.9,
            '美滋滋': 0.8, '笑死': 0.7, '太棒': 0.9, '真好': 0.7, '幸福': 0.8,
            '满足': 0.7, '兴奋': 0.8, '激动': 0.8, '棒': 0.8, '好运': 0.7,
            '涨工资': 0.9, '加薪': 0.9, '升职': 0.9, '成功': 0.8, 'offer': 0.9,
            '过了': 0.7, '获奖': 0.8, '中了': 0.8, '抢到': 0.8, '上线': 0.7,
            '开心': 0.9, '快乐': 0.9, '高兴': 0.9
        },
        'sadness': {
            '难过': 0.9, '伤心': 0.9, '悲伤': 0.9, '痛苦': 0.8, '失落': 0.8,
            '沮丧': 0.8, '郁闷': 0.8, '压抑': 0.7, '崩溃': 0.9,
            '哭': 0.8, '泪': 0.7, '心碎': 0.9, '心痛': 0.8, '凄凉': 0.7,
            '沮丧': 0.8, '心塞': 0.8, '蓝瘦': 0.8, '香菇': 0.8, '丧': 0.7,
            '难受': 0.8, '低落': 0.7, '苦': 0.6, '痛': 0.6, '伤感': 0.7,
            '哀伤': 0.7, '沉重': 0.7, '吵架': 0.8, '分手': 0.9, '离婚': 0.9,
            '背叛': 0.9, '失恋': 0.9, '挂科': 0.8, '失败': 0.7, '没': 0.4
        },
        'anger': {
            '生气': 0.9, '愤怒': 0.9, '气愤': 0.9, '恼火': 0.8, '发怒': 0.9,
            '大怒': 0.9, '火大': 0.9, '气死': 0.9, '可恶': 0.8, '讨厌': 0.7,
            '烦': 0.6, '憎恨': 0.9, '恨': 0.8, '怨': 0.7, '怒': 0.8,
            '烦躁': 0.7, '不爽': 0.6, '坑': 0.8, '太差': 0.9, '画饼': 0.8,
            '被骗': 0.9, '无语': 0.7, '过分': 0.8, '黑心': 0.9, '无良': 0.9,
            '黄牛': 0.7, '奸商': 0.9, '甩锅': 0.8, '瞎指挥': 0.8, '改需求': 0.7,
            '气': 0.6, '真开心': 0.3  # 反讽
        },
        'fear': {
            '害怕': 0.9, '恐惧': 0.9, '担心': 0.7, '担忧': 0.7, '紧张': 0.7,
            '不安': 0.7, '怕': 0.7, '慌': 0.7, '心虚': 0.7, '胆怯': 0.8,
            '畏': 0.7, '恐怖': 0.9, '可怕': 0.8, '惊人': 0.7, '震惊': 0.6,
            '惊慌': 0.9, '恐慌': 0.9, '住院': 0.9, '手术': 0.9, '生病': 0.8,
            '体检': 0.8, '复查': 0.8, '面试': 0.8, '差点': 0.8, '危险': 0.9,
            '骗子': 0.9, '诈骗': 0.9, '盗': 0.9, '地震': 0.9, '车祸': 0.9,
            '走丢': 0.9, '裁员': 0.8, '失业': 0.8
        },
        'anxiety': {
            '焦虑': 0.9, '着急': 0.8, '焦急': 0.8, '心急': 0.8, '忧虑': 0.8,
            '担忧': 0.7, '不安': 0.6, '紧张': 0.6, '忐忑': 0.8, '心慌': 0.8,
            '发慌': 0.8, '愁': 0.7, '压力': 0.8, '失眠': 0.8, 'DDL': 0.9,
            '写不完': 0.8, '进度': 0.7, '落后': 0.7, '不顺利': 0.7, '资金': 0.7,
            '催婚': 0.7, '教育': 0.6, '方向': 0.6, '团队': 0.6, '客户': 0.6
        },
        'surprise': {
            '惊讶': 0.9, '吃惊': 0.9, '震惊': 0.8, '意外': 0.9, '惊奇': 0.9,
            '惊喜': 0.9, '惊': 0.7, '哇': 0.8, '天哪': 0.9, '没想到': 0.9,
            '不敢相信': 0.9, '居然': 0.8, '竟然': 0.8, '出乎意料': 0.9,
            '卖爆': 0.9, '突然': 0.7, '居然': 0.8, '没想到': 0.9
        },
        'love': {
            '喜欢': 0.9, '爱': 0.9, '爱': 0.9, '喜欢': 0.9, '心动': 0.8,
            '甜蜜': 0.9, '甜': 0.7, '浪漫': 0.8, '温馨': 0.8, '温暖': 0.8,
            '温柔': 0.7, '倾心': 0.9, '心仪': 0.8, '中意': 0.8, '爱慕': 0.9,
            '在一起': 0.8, '约会': 0.8, '老婆': 0.6, '老公': 0.6, '孩子': 0.5,
            '家人': 0.5, '宠物': 0.6, '治愈': 0.7, '礼物': 0.6
        },
        'trust': {
            '信任': 0.9, '依赖': 0.8, '依靠': 0.8,
            '托付': 0.9, '放心': 0.9, '信赖': 0.9, '靠谱': 0.95,
            '没问题': 0.95, '相信': 0.85, '靠得住': 0.95
        },
        'anticipation': {
            '期待': 0.9, '希望': 0.8, '盼望': 0.9, '憧憬': 0.9, '展望': 0.8,
            '预料': 0.7, '预测': 0.7, '计划': 0.6, '等着': 0.7, '就要': 0.6,
            '下周': 0.5, '等着': 0.7, '年终奖': 0.7
        },
        'optimism': {
            '乐观': 0.9, '阳光': 0.8, '积极': 0.8, '希望': 0.6, '信心': 0.5,
            '顺利': 0.7, '光明': 0.8, '一切都会': 0.9, '明天会更好': 0.95,
            '困难只是': 0.6, '暂时的': 0.6, '会好起来': 0.7, '曙光': 0.9
        },
        'guilt': {
            '愧疚': 0.9, '自责': 0.9, '抱歉': 0.8, '对不起': 0.9, '过意不去': 0.9,
            '后悔': 0.8, '懊悔': 0.9, '不好意思': 0.7, '应该': 0.4, '早知道': 0.7
        },
        'envy': {
            '羡慕': 0.9, '嫉妒': 0.9, '眼红': 0.8, '不平衡': 0.8, '酸': 0.7,
            '柠檬': 0.8, '馋': 0.7, '别人家的': 0.8, '比我': 0.6, '比他': 0.6,
            '优秀': 0.5
        },
        'contempt': {
            '鄙视': 0.9, '藐视': 0.9, '轻蔑': 0.9, '看不起': 0.9, '不屑': 0.8,
            '渣渣': 0.9, '辣鸡': 0.9, '废物': 0.9, '配': 0.7, '太垃圾': 0.9,
            '差劲': 0.9, '无语': 0.7, '什么人都有': 0.8
        },
        'despair': {
            '绝望': 0.9, '无望': 0.9, '没希望': 0.9, '死心': 0.8, '放弃': 0.8,
            '颓废': 0.9, '崩溃': 0.8, '没救': 0.9, '废了': 0.9, '完了': 0.9,
            '心灰意冷': 0.9, '彻底': 0.6
        },
        'disgust': {
            '恶心': 0.9, '厌恶': 0.9, '讨厌': 0.7, '反感': 0.8, '嫌弃': 0.8,
            '鄙视': 0.7, '轻蔑': 0.7, '不屑': 0.6, '作呕': 0.9, '呕': 0.8,
            '吐': 0.8, '脏': 0.7, '污': 0.7, '虫子': 0.8, '怪味': 0.8,
            '烂': 0.6, '变质': 0.8, '咸猪手': 0.9
        }
    }

    # 特殊模式（高优先级）
    SPECIAL_PATTERNS = [
        # 反讽
        (r'真开心啊', 'anger', 0.9),
        (r'真棒啊', 'anger', 0.9),
        (r'太.*好了', 'anger', 0.8),
        (r'真不错啊', 'anger', 0.8),
        (r'真高兴啊', 'anger', 0.8),
        (r'可真行', 'anger', 0.9),
        # 焦虑/担心 - 考试等特定场景
        (r'担心考试', 'anxiety', 0.95),
        (r'担心.*考不好', 'anxiety', 0.95),
        (r'考试.*紧张', 'anxiety', 0.95),
        # 喜悦 - 正面事件
        (r'涨工资了', 'joy', 0.95),
        (r'加薪了', 'joy', 0.95),
        (r'升职了', 'joy', 0.95),
        (r'成功了', 'joy', 0.9),
        (r'拿到.*offer', 'joy', 0.95),
        (r'考试通过了', 'joy', 0.95),
        (r'获奖了', 'joy', 0.9),
        (r'孩子出生了', 'joy', 0.95),
        (r'收到礼物了', 'joy', 0.9),
        (r'找到新工作了', 'joy', 0.95),
        (r'项目上线了', 'joy', 0.9),
        (r'年终奖发了', 'joy', 0.9),
        (r'好开心', 'joy', 0.9),
        (r'太高兴了', 'joy', 0.9),
        # 惊讶 - 意外事件（可能是正面或负面）
        (r'没想到', 'surprise', 0.95),
        (r'太意外了', 'surprise', 0.95),
        (r'没想到他居然', 'surprise', 0.95),
        (r'突然收到', 'surprise', 0.9),
        (r'面试居然过了', 'surprise', 0.95),
        (r'考试成绩出乎意料', 'surprise', 0.95),
        (r'产品卖爆了', 'surprise', 0.95),
        (r'天哪', 'surprise', 0.95),
        (r'哇', 'surprise', 0.9),
        (r'居然成功了', 'surprise', 0.9),
        (r'居然同意了', 'surprise', 0.9),
        (r'竟然过了', 'surprise', 0.9),
        (r'出人意料', 'surprise', 0.95),
        (r'中了彩票', 'surprise', 0.9),
        # 信任 - 对人或机构的能力/可靠性有信心（需要有具体对象）
        (r'相信他能做好', 'trust', 0.98),
        (r'对这个品牌很信赖', 'trust', 0.98),
        (r'交给他放心', 'trust', 0.98),
        (r'对这个方案有信心', 'trust', 0.98),
        (r'信任这个团队', 'trust', 0.98),
        (r'没问题', 'trust', 0.98),
        (r'靠谱', 'trust', 0.98),
        (r'靠得住', 'trust', 0.98),
        (r'相信会好起来的', 'optimism', 0.99),
        (r'相信.*会', 'trust', 0.98),
        (r'对.*有信心', 'trust', 0.95),
        # 乐观 - 对未来结果积极期待（无具体对象，是一般性陈述）
        (r'一切都会变好的', 'optimism', 0.98),
        (r'明天会更好', 'optimism', 0.98),
        (r'困难只是暂时的', 'optimism', 0.98),
        (r'前景一片光明', 'optimism', 0.98),
        (r'对未来充满信心', 'optimism', 0.95),
        (r'会好起来的', 'optimism', 0.95),
        # 爱 - 亲密关系
        (r'和老婆在一起很幸福', 'love', 0.98),
        (r'老婆怀孕了好开心', 'love', 0.98),
        (r'和女朋友约会', 'love', 0.95),
        (r'孩子太可爱了', 'love', 0.95),
        (r'家人团聚', 'love', 0.95),
        (r'收到.*礼物', 'love', 0.9),
        (r'养了只宠物', 'love', 0.9),
        (r'心都化了', 'love', 0.95),
        (r'好喜欢', 'love', 0.98),
        (r'太可爱了', 'love', 0.9),
        (r'好感人', 'love', 0.95),
        # 绝望/放弃
        (r'绝望了', 'despair', 0.95),
        (r'彻底放弃了', 'despair', 0.95),
        (r'心灰意冷', 'despair', 0.95),
        (r'没救了', 'despair', 0.95),
        (r'彻底完了', 'despair', 0.95),
        (r'完了完了', 'despair', 0.95),
        (r'什么都没意义', 'despair', 0.95),
        (r'没意思', 'despair', 0.9),
        # 乐观/希望
        (r'一切都会变好的', 'optimism', 0.95),
        (r'明天会更好', 'optimism', 0.95),
        (r'困难只是暂时的', 'optimism', 0.95),
        (r'前景一片光明', 'optimism', 0.95),
        (r'对未来充满信心', 'optimism', 0.95),
        # 嫉妒/羡慕
        (r'羡慕别人的', 'envy', 0.98),
        (r'嫉妒他', 'envy', 0.95),
        (r'嫉妒她', 'envy', 0.95),
        (r'别人家的', 'envy', 0.95),
        (r'柠檬精', 'envy', 0.98),
        (r'酸了', 'envy', 0.9),
        (r'我酸了', 'envy', 0.95),
        (r'好羡慕', 'envy', 0.95),
        (r'真羡慕', 'envy', 0.95),
        (r'羡慕不已', 'envy', 0.95),
        (r'他家.*真', 'envy', 0.9),
        (r'同学都.*了', 'envy', 0.9),
        (r'凭什么', 'envy', 0.9),
        (r'比.*好', 'envy', 0.85),
        (r'比我.*好', 'envy', 0.9),
        (r'比他.*好', 'envy', 0.9),
    ]

    def __init__(self):
        self.negation_words = ['不', '没', '无', '非', '别']
        self.intensity_words = ['很', '太', '非常', '特别', '十分', '超级', '巨', '超']

    def detect(self, text: str) -> Optional[MultiLabelEmotionResult]:
        """检测多标签情感"""
        if not text or len(text) < 2:
            return None

        # 初始化各情感得分
        emotion_scores: Dict[str, float] = {}

        # 1. 检查特殊模式
        special_emotions: Set[str] = set()
        for pattern, emotion, confidence in self.SPECIAL_PATTERNS:
            if re.search(pattern, text):
                # 检查否定词
                has_negation = any(neg in text for neg in self.negation_words)
                if has_negation:
                    # 否定反转，但保留一定权重
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + confidence * 0.3
                else:
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + confidence
                special_emotions.add(emotion)

        # 2. 关键词匹配
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if emotion in special_emotions:
                continue  # 特殊模式已处理

            score = 0.0
            for keyword, weight in keywords.items():
                if keyword in text:
                    has_negation = any(neg in text for neg in self.negation_words)
                    has_intensity = any(intens in text for intens in self.intensity_words)

                    # 计算权重
                    final_weight = weight
                    if has_intensity:
                        final_weight *= 1.2
                    if has_negation:
                        # 否定降低权重，但不反转
                        final_weight *= 0.5

                    score += final_weight

            if score > 0:
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

        if not emotion_scores:
            return None

        # 归一化得分
        max_score = max(emotion_scores.values()) if emotion_scores else 1.0
        normalized_scores = {
            k: min(v / max_score, 1.0) for k, v in emotion_scores.items()
        }

        # 排序
        sorted_emotions = sorted(normalized_scores.items(), key=lambda x: -x[1])
        top_emotion, top_confidence = sorted_emotions[0]

        # 判断是否多标签（多个情感都超过阈值）
        threshold = 0.4
        multi_label_emotions = [(e, c) for e, c in sorted_emotions if c >= threshold]
        is_multi_label = len(multi_label_emotions) > 1

        return MultiLabelEmotionResult(
            primary_emotion=top_emotion,
            all_emotions=normalized_scores,
            top_k=sorted_emotions[:5],
            vad=self._emotion_to_vad(top_emotion),
            confidence=top_confidence,
            source='multi_label_rule',
            is_multi_label=is_multi_label
        )

    def _emotion_to_vad(self, emotion: str) -> Tuple[float, float, float]:
        """情感转VAD"""
        vad_map = {
            'joy': (0.8, 0.5, 0.7),
            'sadness': (-0.8, -0.3, -0.5),
            'anger': (-0.8, 0.7, 0.5),
            'fear': (-0.6, 0.6, -0.4),
            'anxiety': (-0.5, 0.6, -0.4),
            'love': (0.9, 0.4, 0.8),
            'disgust': (-0.7, -0.1, -0.4),
            'surprise': (0.3, 0.8, 0.3),
            'despair': (-0.9, 0.3, -0.7),
            'contempt': (-0.6, 0.2, 0.4),
            'envy': (-0.4, 0.3, -0.3),
            'guilt': (-0.5, 0.2, -0.4),
            'trust': (0.6, 0.3, 0.7),
            'anticipation': (0.5, 0.6, 0.4),
            'optimism': (0.7, 0.5, 0.6),
        }
        return vad_map.get(emotion, (0, 0, 0))


if __name__ == "__main__":
    detector = MultiLabelEmotionDetector()

    test_texts = [
        "今天涨工资了，太开心了！",
        "被裁员了，不知道怎么办",
        "产品质量太差了，坑人",
        "又失业又下雨，真倒霉",
        "虽然加班很累，但项目上线了很开心",
    ]

    print("=== Multi-Label Emotion Detection ===")
    for text in test_texts:
        result = detector.detect(text)
        if result:
            print(f"\nText: {text}")
            print(f"Primary: {result.primary_emotion} ({result.confidence:.2f})")
            print(f"Top emotions: {result.top_k}")
            print(f"Is multi-label: {result.is_multi_label}")