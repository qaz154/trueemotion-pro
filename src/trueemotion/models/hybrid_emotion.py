# -*- coding: utf-8 -*-
"""
Hybrid Emotion Detector - Neural + Rule-based
神经网络 + 规则系统的混合情感检测
"""

import re
from typing import Dict, List, Tuple, Optional


class RuleBasedEmotionDetector:
    """基于规则的情感检测器"""

    # 情感关键词
    EMOTION_KEYWORDS = {
        'joy': ['开心', '高兴', '快乐', '欢乐', '喜悦', '欢喜', '愉快', '开心', '快乐', '爽', '耶', '哈哈', '哈哈哈', '美滋滋', '笑死', '太棒', '真好', '真好', '开心', '快乐', '幸福', '满足', '兴奋', '激动', '棒', '好运'],
        'sadness': ['难过', '伤心', '悲伤', '痛苦', '悲伤', '失落', '沮丧', '郁闷', '压抑', '绝望', '崩溃', '哭', '泪', '心碎', '心痛', '凄凉', '凄然', '沮丧', '失落', '心塞', '蓝瘦', '香菇', '丧', '难受', '难过', '低落', '苦', '痛', '伤感', '哀伤', '沉重', '压抑', '失落', '累', '身心俱疲'],
        'anger': ['生气', '愤怒', '气愤', '恼火', '发怒', '大怒', '火大', '气死', '气死了', '可恶', '讨厌', '烦', '讨厌', '憎恨', '恨', '怨', '怒', '气愤', '恼怒', '烦躁', '烦躁', '不爽'],
        'fear': ['害怕', '恐惧', '担心', '担忧', '紧张', '不安', '怕', '慌', '心虚', '胆怯', '畏', '怕', '恐怖', '可怕', '惊人', '震惊', '惊慌', '恐慌'],
        'anxiety': ['焦虑', '着急', '焦急', '心急', '焦虑', '忧虑', '担忧', '不安', '紧张', '忐忑', '心慌', '发慌', '怕', '担心', '愁', '烦', '烦躁', '压力', '失眠', '加班', '疲惫'],
        'love': ['喜欢', '爱', '爱', '喜欢', '心动', '甜蜜', '甜', '浪漫', '温馨', '温暖', '温柔', '倾心', '心仪', '中意', '爱慕', 'love'],
        'disgust': ['恶心', '厌恶', '讨厌', '反感', '嫌弃', '鄙视', '轻蔑', '不屑', '作呕', '呕', '吐', '脏', '污', '讨厌', '反感', '厌恶'],
        'surprise': ['惊讶', '吃惊', '震惊', '意外', '惊奇', '惊讶', '意外', '惊喜', '惊', '哇', '天哪', '没想到', '不敢相信'],
        'despair': ['绝望', '无望', '没希望', '绝望', '死心', '放弃', '颓废', '失落', '崩溃', '绝望', '没救', '废了', '完了'],
        'contempt': ['鄙视', '藐视', '轻蔑', '看不起', '不屑', '藐视', '鄙视', '轻蔑', '看不起', '渣渣', '辣鸡', '废物'],
        'envy': ['羡慕', '嫉妒', '眼红', '酸', '柠檬', '不平衡', '羡慕', '嫉妒', '眼红', '馋'],
        'guilt': ['愧疚', '自责', '抱歉', '对不起', '过意不去', '愧疚', '自责', '抱歉', '对不起', '抱歉', '不好意思'],
    }

    # 否定词（反转情感）
    NEGATION_WORDS = ['不', '没', '无', '非', '别', '不是', '没是', '不会', '不能', '无法', '不会', '不能']

    # 程度词（增强情感）
    INTENSITY_WORDS = ['很', '太', '非常', '特别', '十分', '超级', '巨', '超', '极其', '格外', '真', '好']

    # 复合否定模式（"不太"、"没太"等反转）
    NEGATION_PATTERNS = [
        (r'不太', 'reverse'),
        (r'没太', 'reverse'),
        (r'不是太', 'reverse'),
        (r'不太', 'reverse'),
    ]

    # 特殊情感短语（优先匹配）
    SPECIAL_PATTERNS = [
        # 反讽类（用正面词表达负面情感）
        (r'真开心啊', 'anger'),   # 反讽：实际是生气/不满
        (r'真棒啊', 'anger'),     # 反讽
        (r'太好了', 'anger'),     # 反讽（当语境明显负面时）
        (r'真不错啊', 'anger'),   # 反讽
        (r'真高兴啊', 'anger'),   # 反讽
        (r'真棒', 'anger'),       # 反讽（语气词）
        (r'真行啊', 'anger'),     # 反讽：你可真行（贬义）
        (r'可真', 'anger'),       # 反讽：你可真行
        # 愤怒类
        (r'太.*了.*坑', 'anger'),
        (r'太差了', 'anger'),
        (r'太气人了', 'anger'),
        (r'太过分了', 'anger'),
        (r'太不像话了', 'anger'),
        (r'被骗了', 'anger'),
        (r'被.*骂了', 'anger'),  # 被老板骂了、被领导骂了等
        (r'被背叛', 'anger'),
        (r'被坑了', 'anger'),
        (r'被甩了', 'sadness'),
        (r'太不公平', 'anger'),
        (r'黑心', 'anger'),
        (r'无良', 'anger'),
        (r'画饼', 'anger'),  # 画饼 = 空头承诺，令人愤怒
        # 悲伤类
        (r'太难过了', 'sadness'),
        (r'太伤心', 'sadness'),
        (r'太难受了', 'sadness'),
        (r'太痛苦', 'sadness'),
        (r'发烧.*天', 'sadness'),
        (r'感冒.*天', 'sadness'),
        (r'生病了', 'fear'),
        (r'住院', 'fear'),
        (r'手术', 'fear'),
        (r'挂了.*科', 'sadness'),
        (r'挂科', 'sadness'),
        # 悲伤类 - 关系破裂
        (r'吵架', 'sadness'),
        (r'分手', 'sadness'),
        (r'离婚', 'sadness'),
        (r'背叛', 'sadness'),
        (r'被.*背叛', 'sadness'),
        (r'失恋', 'sadness'),
        (r'失业', 'fear'),
        # 绝望类
        (r'没有希望', 'despair'),
        (r'没希望', 'despair'),
        (r'人生没有希望', 'despair'),
        (r'彻底绝望', 'despair'),
        (r'裁员', 'fear'),
        # 焦虑类
        (r'赶飞机', 'anxiety'),
        (r'要面试', 'fear'),
        (r'马上.*面试', 'fear'),
        (r'面试.*紧张', 'fear'),
        (r'要考试', 'anxiety'),
        (r'要汇报', 'anxiety'),
        (r'要演讲', 'anxiety'),
        (r'担心', 'anxiety'),
        (r'害怕', 'fear'),
        (r'压力.*大', 'anxiety'),
        (r'睡不着', 'anxiety'),
        (r'紧张', 'anxiety'),
        # 恐惧类 - 健康/危险
        (r'住院', 'fear'),
        (r'手术', 'fear'),
        (r'生病了', 'fear'),
        (r'体检', 'fear'),
        (r'检查.*病', 'fear'),
        # 喜悦类
        (r'涨工资', 'joy'),
        (r'加薪', 'joy'),
        (r'升职', 'joy'),
        (r'成功了', 'joy'),
        (r'拿到.*offer', 'joy'),
        (r'考研.*过', 'joy'),
        (r'考试.*过', 'joy'),
        (r'获奖了', 'joy'),
        (r'中彩票', 'joy'),
        (r'抢到.*票', 'joy'),
        # 学习/考试类
        (r'没考好', 'sadness'),
        (r'考.*失败', 'sadness'),
        (r'感觉自己.*失败', 'sadness'),
        # 健康类
        (r'体检.*不好', 'fear'),
        (r'体检结果', 'fear'),
        (r'进一步检查', 'fear'),
        # 工作机会
        (r'有工作了', 'joy'),
        (r'找到工作了', 'joy'),
        (r'入职', 'joy'),
        # 项目/工作完成
        (r'上线了', 'joy'),
        (r'项目.*上线', 'joy'),
        (r'完成了.*项目', 'joy'),
        # 惊讶类
        (r'没想到', 'surprise'),
        (r'居然', 'surprise'),
        (r'竟然', 'surprise'),
        (r'意外', 'surprise'),
        (r'吃惊', 'surprise'),
        (r'震惊', 'surprise'),
        (r'哇', 'surprise'),
        (r'天哪', 'surprise'),
        (r'出人意料', 'surprise'),
        (r'没想到', 'surprise'),
        (r'突然.*收到', 'surprise'),
        (r'居然.*过了', 'surprise'),
        (r'出乎意料', 'surprise'),
        (r'卖爆了', 'surprise'),
        # 嫉妒/羡慕类
        (r'羡慕', 'envy'),
        (r'嫉妒', 'envy'),
        (r'眼红', 'envy'),
        (r'别人家的', 'envy'),
        (r'比我.*好', 'envy'),
        (r'比他.*优秀', 'envy'),
        # 恐惧类 - 危险/威胁
        (r'差点', 'fear'),
        (r'危险', 'fear'),
        (r'骗子', 'fear'),
        (r'被盗', 'fear'),
        (r'诈骗', 'fear'),
        (r'地震', 'fear'),
        (r'车祸', 'fear'),
        (r'走丢', 'fear'),
        (r'挂科了', 'fear'),
    ]

    def __init__(self):
        """初始化检测器"""
        self.learned_patterns = []  # 学习到的模式

    def add_learned_pattern(self, pattern: str, emotion: str):
        """添加学习到的模式"""
        if pattern and emotion and (pattern, emotion) not in self.learned_patterns:
            self.learned_patterns.append((pattern, emotion))

    def clear_learned_patterns(self):
        """清除学习到的模式"""
        self.learned_patterns = []

    def detect(self, text: str) -> Optional[Dict]:
        """检测文本情感"""
        if not text or len(text) < 2:
            return None

        text_lower = text.lower()

        # 1. 先检查特殊模式（优先匹配）- 包括学习到的模式
        all_patterns = list(self.SPECIAL_PATTERNS) + self.learned_patterns
        for pattern, emotion in all_patterns:
            if re.search(pattern, text):
                return {
                    'emotion': emotion,
                    'confidence': 0.85,
                    'matched_pattern': pattern,
                    'has_negation': False,
                    'has_intensity': False,
                    'is_learned': pattern in [p for p, _ in self.learned_patterns]
                }

        # 2. 统计每个情感的匹配次数
        emotion_scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score

        if not emotion_scores:
            return None

        # 找最高分的情感
        best_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])

        # 计算置信度
        total_score = sum(emotion_scores.values())
        confidence = emotion_scores[best_emotion] / total_score if total_score > 0 else 0

        # 检测否定（可能反转情感）
        has_negation = any(neg in text for neg in self.NEGATION_WORDS)

        # 检测程度
        has_intensity = any(intens in text for intens in self.INTENSITY_WORDS)

        return {
            'emotion': best_emotion,
            'confidence': min(confidence * (1.2 if has_intensity else 1.0), 1.0),
            'scores': emotion_scores,
            'has_negation': has_negation,
            'has_intensity': has_intensity
        }


class HybridEmotionAnalyzer:
    """
    混合情感分析器
    - 高置信度时使用神经网络
    - 低置信度时使用规则系统
    - 综合两者结果
    - 支持学习到的模式
    """

    def __init__(self, neural_model_path: str = None):
        self.rule_detector = RuleBasedEmotionDetector()
        self.neural_analyzer = None

        if neural_model_path:
            try:
                from trueemotion.models.char_emotion_model import CharEmotionAnalyzer
                self.neural_analyzer = CharEmotionAnalyzer()
                self.neural_analyzer.load(neural_model_path)
                print(f"Neural model loaded: {neural_model_path}")
            except Exception as e:
                print(f"Failed to load neural model: {e}")

    def add_learned_pattern(self, pattern: str, emotion: str):
        """添加学习到的模式"""
        self.rule_detector.add_learned_pattern(pattern, emotion)

    def load_patterns_from_dict(self, patterns: Dict):
        """从字典加载模式"""
        for pattern, info in patterns.items():
            if isinstance(info, dict) and 'emotion' in info:
                self.add_learned_pattern(pattern, info['emotion'])

    def analyze(self, text: str) -> Dict:
        """分析文本情感"""

        # 1. 先用规则系统（特殊模式优先）
        rule_result = self.rule_detector.detect(text)

        # 2. 用神经网络
        neural_result = None
        if self.neural_analyzer:
            results = self.neural_analyzer.predict([text])
            if results:
                r = results[0]
                neural_result = {
                    'emotion': r['primary_emotion'],
                    'confidence': r['primary_score'],
                    'vad': r['vad'],
                    'source': 'neural'
                }

        # 3. 综合判断 - 规则特殊模式优先
        if rule_result and 'matched_pattern' in rule_result:
            # 规则系统匹配到特殊模式，高优先级使用
            if neural_result:
                return {
                    'primary_emotion': rule_result['emotion'],
                    'confidence': rule_result['confidence'],
                    'vad': self._emotion_to_vad(rule_result['emotion']),
                    'source': 'rule_special',
                    'neural_emotion': neural_result['emotion'],
                    'pattern': rule_result.get('matched_pattern')
                }
            else:
                return {
                    'primary_emotion': rule_result['emotion'],
                    'confidence': rule_result['confidence'],
                    'vad': self._emotion_to_vad(rule_result['emotion']),
                    'source': 'rule_special_only'
                }

        # 4. 非特殊规则结果 + 神经网络
        if rule_result and neural_result:
            # 两者都有结果
            rule_conf = rule_result['confidence']
            neural_conf = neural_result['confidence']
            rule_emotion = rule_result['emotion']
            neural_emotion = neural_result['emotion']

            # 规则高置信度（>=0.7）优先使用规则
            if rule_conf >= 0.7:
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': rule_conf,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'rule_high_conf',
                    'neural_emotion': neural_emotion,
                    'neural_conf': neural_conf
                }

            # 规则和神经网络一致，融合结果
            if rule_emotion == neural_emotion:
                # 一致时，规则 boosting
                boosted_conf = min(rule_conf * 1.2, 1.0)
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': boosted_conf,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'rule_neural_agree',
                    'rule_conf': rule_conf,
                    'neural_conf': neural_conf
                }

            # 规则和神经网络不一致
            if neural_conf < 0.5:
                # 神经网络低置信度，使用规则
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': rule_conf * 0.9,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'rule_override_low_neural',
                    'neural_emotion': neural_emotion,
                    'neural_conf': neural_conf
                }
            elif rule_conf >= 0.5:
                # 规则也有一定置信度，使用规则（更安全）
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': rule_conf,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'rule_override',
                    'neural_emotion': neural_emotion,
                    'neural_conf': neural_conf
                }
            else:
                # 都低置信度，神经网络为主但降低置信度
                return {
                    'primary_emotion': neural_emotion,
                    'confidence': neural_conf * 0.8,
                    'vad': neural_result['vad'],
                    'source': 'hybrid_disagree',
                    'rule_emotion': rule_emotion,
                    'rule_conf': rule_conf
                }

        elif neural_result:
            return {
                'primary_emotion': neural_result['emotion'],
                'confidence': neural_result['confidence'],
                'vad': neural_result['vad'],
                'source': 'neural_only'
            }

        elif rule_result:
            return {
                'primary_emotion': rule_result['emotion'],
                'confidence': rule_result['confidence'] * 0.8,
                'vad': self._emotion_to_vad(rule_result['emotion']),
                'source': 'rule_only'
            }

        return None

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
        }
        return vad_map.get(emotion, (0, 0, 0))


if __name__ == "__main__":
    analyzer = HybridEmotionAnalyzer('../models/v4_model.pt')

    tests = [
        '今天加班到很晚，但项目终于上线了',
        '被裁员了，不知道怎么办',
        '涨工资了！',
        '这个产品质量太差了，坑人',
        '考研成绩过了',
    ]

    print("Hybrid Emotion Analysis:")
    print("=" * 50)
    for text in tests:
        result = analyzer.analyze(text)
        print(f"{text[:20]:<20} -> {result['primary_emotion']:<10} (conf:{result['confidence']:.2f}) [{result['source']}]")
