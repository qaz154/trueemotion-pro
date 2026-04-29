# -*- coding: utf-8 -*-
"""
BERT-based Emotion Analyzer
使用预训练中文BERT模型进行情感分析
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Dict, List, Optional, Tuple
import re


class BertEmotionClassifier(nn.Module):
    """BERT情感分类器"""

    def __init__(self, model_name: str = "bert-base-chinese", num_emotions: int = 15, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # 情感分类头
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

        # VAD回归头
        self.vad_predictor = nn.Linear(self.bert.config.hidden_size, 3)

        # 强度回归头
        self.intensity_predictor = nn.Linear(self.bert.config.hidden_size, 1)

        self.num_emotions = num_emotions

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用[CLS] token的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 情感 logits
        emotion_logits = self.emotion_classifier(pooled_output)

        # VAD预测
        vad_pred = torch.tanh(self.vad_predictor(pooled_output))

        # 强度预测
        intensity_pred = torch.sigmoid(self.intensity_predictor(pooled_output)).squeeze(-1)

        return {
            "emotion_logits": emotion_logits,
            "vad_pred": vad_pred,
            "intensity_pred": intensity_pred
        }


class BertEmotionAnalyzer:
    """
    BERT情感分析器

    支持15种情感分类 + VAD维度 + 强度预测
    """

    # 情感标签
    EMOTION_LABELS = [
        'anger',        # 愤怒
        'anticipation', # 期待
        'anxiety',      # 焦虑
        'contempt',     # 鄙视
        'despair',      # 绝望
        'disgust',      # 厌恶
        'envy',         # 嫉妒
        'fear',         # 恐惧
        'guilt',        # 内疚
        'joy',          # 喜悦
        'love',         # 爱
        'optimism',     # 乐观
        'sadness',      # 悲伤
        'surprise',     # 惊讶
        'trust',        # 信任
    ]

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_path = model_path

        # VAD映射
        self.VAD_MAP = {
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

    def load(self, model_path: str):
        """加载模型"""
        self.model_path = model_path

        # 加载tokenizer
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # 加载模型
        self.model = BertEmotionClassifier(
            model_name="bert-base-chinese",
            num_emotions=15
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"BERT model loaded from {model_path}")

    def predict(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """预测情感"""
        if self.model is None:
            return []

        results = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # 预测
                outputs = self.model(input_ids, attention_mask)

                # 处理每个结果
                probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()
                vad_preds = outputs["vad_pred"].cpu().numpy()
                intensity_preds = outputs["intensity_pred"].cpu().numpy()

                for j, text in enumerate(batch_texts):
                    pred_idx = probs[j].argmax()
                    pred_emotion = self.EMOTION_LABELS[pred_idx]
                    confidence = probs[j].max()

                    results.append({
                        "text": text,
                        "primary_emotion": pred_emotion,
                        "primary_score": float(confidence),
                        "vad": tuple(vad_preds[j]),
                        "intensity": float(intensity_preds[j]) if j < len(intensity_preds) else 0.5,
                        "all_emotions": {
                            self.EMOTION_LABELS[k]: float(probs[j][k])
                            for k in range(len(self.EMOTION_LABELS))
                        }
                    })

        return results

    def analyze(self, text: str) -> Dict:
        """分析单条文本"""
        results = self.predict([text])
        if results:
            return results[0]
        return None


class BertRuleHybridAnalyzer:
    """
    BERT + 规则混合分析器

    结合BERT的泛化能力和规则的精确性
    """

    def __init__(self, model_path: str = None):
        from trueemotion.models.hybrid_emotion import RuleBasedEmotionDetector

        self.rule_detector = RuleBasedEmotionDetector()
        self.bert_analyzer = None

        if model_path:
            try:
                self.bert_analyzer = BertEmotionAnalyzer()
                self.bert_analyzer.load(model_path)
                print(f"BERT model loaded: {model_path}")
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
                self.bert_analyzer = None

    def analyze(self, text: str) -> Dict:
        """混合分析"""
        # 1. 规则系统（特殊模式优先）
        rule_result = self.rule_detector.detect(text)

        # 2. BERT模型
        bert_result = None
        if self.bert_analyzer:
            results = self.bert_analyzer.predict([text])
            if results:
                bert_result = results[0]

        # 3. 综合判断
        if rule_result and 'matched_pattern' in rule_result:
            # 规则特殊模式优先
            return {
                'primary_emotion': rule_result['emotion'],
                'confidence': rule_result['confidence'],
                'vad': self._emotion_to_vad(rule_result['emotion']),
                'source': 'rule_special',
                'bert_emotion': bert_result['primary_emotion'] if bert_result else None,
                'pattern': rule_result.get('matched_pattern')
            }

        # 4. 两者都有结果
        if rule_result and bert_result:
            rule_conf = rule_result['confidence']
            bert_conf = bert_result['primary_score']
            rule_emotion = rule_result['emotion']
            bert_emotion = bert_result['primary_emotion']

            # 一致时boost
            if rule_emotion == bert_emotion:
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': max(rule_conf, bert_conf) * 1.1,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'bert_rule_agree',
                    'rule_conf': rule_conf,
                    'bert_conf': bert_conf
                }

            # 规则高置信度
            if rule_conf >= 0.7:
                return {
                    'primary_emotion': rule_emotion,
                    'confidence': rule_conf,
                    'vad': self._emotion_to_vad(rule_emotion),
                    'source': 'rule_override',
                    'bert_emotion': bert_emotion,
                    'bert_conf': bert_conf
                }

            # BERT高置信度
            if bert_conf >= 0.7:
                return {
                    'primary_emotion': bert_emotion,
                    'confidence': bert_conf,
                    'vad': bert_result['vad'],
                    'source': 'bert_override',
                    'rule_emotion': rule_emotion
                }

            # 都低置信度，参考BERT
            return {
                'primary_emotion': bert_emotion,
                'confidence': bert_conf * 0.9,
                'vad': bert_result['vad'],
                'source': 'bert_fallback',
                'rule_emotion': rule_emotion
            }

        # 5. 只有BERT
        if bert_result:
            return {
                'primary_emotion': bert_result['primary_emotion'],
                'confidence': bert_result['primary_score'],
                'vad': bert_result['vad'],
                'source': 'bert_only'
            }

        # 6. 只有规则
        if rule_result:
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
            'trust': (0.6, 0.3, 0.7),
            'anticipation': (0.5, 0.6, 0.4),
            'optimism': (0.7, 0.5, 0.6),
        }
        return vad_map.get(emotion, (0, 0, 0))


if __name__ == "__main__":
    # 测试BERT分析器
    analyzer = BertEmotionAnalyzer()

    test_texts = [
        "今天涨工资了，太开心了！",
        "被裁员了，不知道怎么办",
        "产品质量太差了，坑人"
    ]

    print("BERT Emotion Analysis:")
    for text in test_texts:
        result = analyzer.analyze(text)
        if result:
            print(f"{text[:15]} -> {result['primary_emotion']} ({result['primary_score']:.2f})")