# -*- coding: utf-8 -*-
"""
TrueEmotion Neural Model - 真正的神经网络情感分析模型
=====================================================

多任务深度学习模型，同时预测：
1. 原型情感分类（8类）
2. 复合情感多标签（16类）
3. VAD维度回归（3个连续值）
4. 情感强度回归（1个连续值）
5. 反讽检测（二分类）

架构：Embedding → BiLSTM → Multi-task Heads
"""

import os
import sys
import re
import math
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trueemotion.emotion.plutchik24 import (
    EMOTION_DEFINITIONS, VAD_LEXICON, get_primary_emotions,
    get_complex_emotions, get_all_emotions
)


# ==================== 配置 ====================

@dataclass
class NeuralModelConfig:
    """神经网络模型配置"""
    # 词向量维度
    embedding_dim: int = 128
    # 隐藏层维度
    hidden_dim: int = 256
    # BiLSTM层数
    num_layers: int = 2
    # Dropout概率
    dropout: float = 0.3
    # 注意力维度
    attention_dim: int = 128

    # 训练参数
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50

    # 损失权重
    emotion_weight: float = 1.0
    vad_weight: float = 0.3
    intensity_weight: float = 0.5
    irony_weight: float = 0.8

    # 情感类别数
    num_primary: int = 8
    num_complex: int = 18  # 实际有18种复合情感
    num_all: int = 26     # 8 + 18 = 26


# ==================== 字符级Tokenizer ====================

class CharTokenizer:
    """简单字符级分词器"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: List[str]):
        """从文本学习词表"""
        chars = set()
        for text in texts:
            chars.update(text)
        for i, char in enumerate(sorted(chars), start=2):
            self.char2idx[char] = i
            self.idx2char[i] = char

    def encode(self, text: str, max_len: int = 64) -> np.ndarray:
        """编码文本为索引序列"""
        result = []
        for char in text[:max_len]:
            result.append(self.char2idx.get(char, 1))
        # Padding
        while len(result) < max_len:
            result.append(0)
        return np.array(result, dtype=np.int64)

    def decode(self, indices: np.ndarray) -> str:
        """解码回文本"""
        return "".join(self.idx2char.get(i, "<UNK>") for i in indices)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"char2idx": self.char2idx, "idx2char": self.idx2char}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.char2idx = data["char2idx"]
            self.idx2char = data["idx2char"]


# ==================== 位置编码 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 注意力层 ====================

class Attention(nn.Module):
    """自注意力层"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> (batch, hidden_dim)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


# ==================== 神经网络模型 ====================

class EmotionNeuralModel(nn.Module):
    """
    多任务情感分析神经网络

    输入: 文本索引序列 (batch, seq_len)
    输出:
      - emotion_logits: (batch, 24) 情感分类
      - vad_pred: (batch, 3) VAD回归
      - intensity_pred: (batch, 1) 强度回归
      - irony_pred: (batch, 1) 反讽检测
    """

    def __init__(self, config: NeuralModelConfig, vocab_size: int):
        super().__init__()
        self.config = config

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)

        # 位置编码
        self.pos_encoder = PositionalEncoding(config.embedding_dim, dropout=config.dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        # 注意力
        self.attention = Attention(config.hidden_dim * 2)

        # 输出层
        hidden_dim = config.hidden_dim * 2

        # 情感分类头 (26类)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 26)  # 26类情感
        )

        # VAD回归头
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

        # 强度回归头
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 反讽检测头
        self.irony_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, seq_len)

        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # 位置编码
        embedded = self.pos_encoder(embedded)

        # BiLSTM
        lstm_output, _ = self.lstm(embedded)
        # lstm_output: (batch, seq_len, hidden_dim * 2)

        # 注意力
        context, attention_weights = self.attention(lstm_output)
        # context: (batch, hidden_dim * 2)

        # 多任务输出
        emotion_logits = self.emotion_head(context)
        vad_pred = torch.tanh(self.vad_head(context))  # VAD范围 [-1, 1]
        intensity_pred = self.intensity_head(context)  # [0, 1]
        irony_pred = torch.sigmoid(self.irony_head(context))  # [0, 1]

        return {
            "emotion_logits": emotion_logits,
            "vad_pred": vad_pred,
            "intensity_pred": intensity_pred,
            "irony_pred": irony_pred,
            "attention_weights": attention_weights
        }


# ==================== 损失函数 ====================

class MultiTaskLoss(nn.Module):
    """多任务损失"""

    def __init__(self, config: NeuralModelConfig):
        super().__init__()
        self.config = config
        # 24类情感使用BCEWithLogitsLoss (多标签分类)
        self.emotion_criterion = nn.BCEWithLogitsLoss()
        # VAD使用MSE
        self.vad_criterion = nn.MSELoss()
        # 强度使用MSE
        self.intensity_criterion = nn.MSELoss()
        # 反讽使用BCE
        self.irony_criterion = nn.BCELoss()

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 情感损失 (多标签BCE)
        emotion_loss = self.emotion_criterion(
            outputs["emotion_logits"],
            targets["emotion_labels"]
        )

        # VAD损失
        vad_loss = self.vad_criterion(
            outputs["vad_pred"],
            targets["vad_labels"]
        )

        # 强度损失
        intensity_loss = self.intensity_criterion(
            outputs["intensity_pred"],
            targets["intensity_labels"]
        )

        # 反讽损失
        irony_loss = self.irony_criterion(
            outputs["irony_pred"],
            targets["irony_labels"]
        )

        # 加权总损失
        total_loss = (
            self.config.emotion_weight * emotion_loss +
            self.config.vad_weight * vad_loss +
            self.config.intensity_weight * intensity_loss +
            self.config.irony_weight * irony_loss
        )

        return {
            "total_loss": total_loss,
            "emotion_loss": emotion_loss,
            "vad_loss": vad_loss,
            "intensity_loss": intensity_loss,
            "irony_loss": irony_loss
        }


# ==================== 数据集 ====================

class EmotionDataset(Dataset):
    """情感数据集"""

    def __init__(self, texts: List[str], labels: Dict[str, np.ndarray], tokenizer: CharTokenizer,
                 max_len: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text, self.max_len)

        return {
            "input_ids": torch.tensor(encoding, dtype=torch.long),
            "emotion_labels": torch.tensor(self.labels["emotion"][idx], dtype=torch.float),
            "vad_labels": torch.tensor(self.labels["vad"][idx], dtype=torch.float),
            "intensity_labels": torch.tensor(self.labels["intensity"][idx], dtype=torch.float),
            "irony_labels": torch.tensor(self.labels["irony"][idx], dtype=torch.float)
        }


# ==================== 模型包装器 ====================

class EmotionNeuralAnalyzer:
    """
    神经网络情感分析器

    提供训练和推理接口
    """

    # 情感名称列表 (按EMOTION_DEFINITIONS顺序)
    EMOTION_NAMES = [
        # 8 primary (0-7)
        "joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust",
        # 16 complex (8-23)
        "optimism", "love", "guilt", "submission", "surprise_complex", "disappointment",
        "remorse", "envy", "suspicion", "aggression", "pride", "contentment", "contempt",
        "cynicism", "morbidness", "sentimentality", "anxiety", "despair"
    ]

    def __init__(self, config: Optional[NeuralModelConfig] = None):
        self.config = config or NeuralModelConfig()
        self.tokenizer = CharTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_tokenizer(self, texts: List[str]):
        """构建词表"""
        self.tokenizer.fit(texts)
        print(f"词表大小: {len(self.tokenizer.char2idx)}")

    def build_model(self):
        """构建模型"""
        vocab_size = len(self.tokenizer.char2idx)
        self.model = EmotionNeuralModel(self.config, vocab_size)
        self.model.to(self.device)
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_data(self, texts: List[str], emotion_labels: np.ndarray,
                     vad_labels: np.ndarray, intensity_labels: np.ndarray,
                     irony_labels: np.ndarray) -> DataLoader:
        """准备数据加载器"""
        labels = {
            "emotion": emotion_labels,
            "vad": vad_labels,
            "intensity": intensity_labels,
            "irony": irony_labels
        }
        dataset = EmotionDataset(texts, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def train_step(self, dataloader: DataLoader) -> Dict[str, float]:
        """单轮训练"""
        self.model.train()
        criterion = MultiTaskLoss(self.config)
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        total_loss = 0
        loss_details = {"emotion": 0, "vad": 0, "intensity": 0, "irony": 0}

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = {
                "emotion_labels": batch["emotion_labels"].to(self.device),
                "vad_labels": batch["vad_labels"].to(self.device),
                "intensity_labels": batch["intensity_labels"].to(self.device),
                "irony_labels": batch["irony_labels"].to(self.device)
            }

            optimizer.zero_grad()
            outputs = self.model(input_ids)
            losses = criterion(outputs, targets)

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += losses["total_loss"].item()
            loss_details["emotion"] += losses["emotion_loss"].item()
            loss_details["vad"] += losses["vad_loss"].item()
            loss_details["intensity"] += losses["intensity_loss"].item()
            loss_details["irony"] += losses["irony_loss"].item()

        num_batches = len(dataloader)
        return {
            "total": total_loss / num_batches,
            "emotion": loss_details["emotion"] / num_batches,
            "vad": loss_details["vad"] / num_batches,
            "intensity": loss_details["intensity"] / num_batches,
            "irony": loss_details["irony"] / num_batches
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        criterion = MultiTaskLoss(self.config)

        total_loss = 0
        emotion_correct = 0
        vad_mae = 0
        intensity_mae = 0
        irony_correct = 0
        total_samples = 0

        primary_indices = list(range(8))  # 前8个是原始情感

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = {
                "emotion_labels": batch["emotion_labels"].to(self.device),
                "vad_labels": batch["vad_labels"].to(self.device),
                "intensity_labels": batch["intensity_labels"].to(self.device),
                "irony_labels": batch["irony_labels"].to(self.device)
            }

            outputs = self.model(input_ids)
            losses = criterion(outputs, targets)

            total_loss += losses["total_loss"].item()

            # 情感准确率 (只看primary emotion)
            emotion_probs = torch.sigmoid(outputs["emotion_logits"])
            pred_primary = emotion_probs[:, primary_indices].argmax(dim=1)
            true_primary = batch["emotion_labels"][:, primary_indices].argmax(dim=1)
            emotion_correct += (pred_primary == true_primary).sum().item()

            # VAD MAE
            vad_mae += F.l1_loss(outputs["vad_pred"], targets["vad_labels"]).item() * 3

            # 强度 MAE
            intensity_mae += F.l1_loss(outputs["intensity_pred"], targets["intensity_labels"]).item()

            # 反讽准确率
            irony_pred = (outputs["irony_pred"] > 0.5).float()
            irony_correct += (irony_pred == targets["irony_labels"]).sum().item()

            total_samples += input_ids.size(0)

        num_batches = len(dataloader)
        return {
            "loss": total_loss / num_batches,
            "emotion_acc": emotion_correct / total_samples * 100,
            "vad_mae": vad_mae / num_batches,
            "intensity_mae": intensity_mae / num_batches,
            "irony_acc": irony_correct / total_samples * 100
        }

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 128) -> List[Dict[str, Any]]:
        """预测"""
        self.model.eval()

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = [self.tokenizer.encode(t) for t in batch_texts]
            input_ids = torch.tensor(encodings, dtype=torch.long).to(self.device)

            outputs = self.model(input_ids)

            emotion_probs = torch.sigmoid(outputs["emotion_logits"]).cpu().numpy()
            vad_preds = outputs["vad_pred"].cpu().numpy()
            intensity_preds = outputs["intensity_pred"].cpu().numpy()
            irony_preds = outputs["irony_pred"].cpu().numpy()

            for j in range(len(batch_texts)):
                # 获取主要情感
                primary_scores = emotion_probs[j, :8]
                primary_idx = primary_scores.argmax()
                primary_emotion = self.EMOTION_NAMES[primary_idx]

                # 获取高置信度复合情感
                complex_scores = emotion_probs[j, 8:]
                complex_emotions = []
                for k, score in enumerate(complex_scores):
                    if score > 0.4:
                        complex_emotions.append(self.EMOTION_NAMES[8 + k])

                results.append({
                    "text": batch_texts[j],
                    "primary_emotion": primary_emotion,
                    "primary_score": float(primary_scores[primary_idx]),
                    "all_emotions": {self.EMOTION_NAMES[k]: float(emotion_probs[j, k])
                                    for k in range(26)},
                    "complex_emotions": complex_emotions,
                    "vad": tuple(float(v) for v in vad_preds[j]),
                    "intensity": float(intensity_preds[j]),
                    "irony_prob": float(irony_preds[j]),
                    "is_irony": bool(irony_preds[j] > 0.5)
                })

        return results

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "tokenizer": self.tokenizer
        }, path)
        print(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.tokenizer = checkpoint["tokenizer"]
        self.build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        print(f"模型已加载: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("TrueEmotion Neural Model - 神经网络情感分析模型")
    print("=" * 70)

    # 测试模型构建
    config = NeuralModelConfig(
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        batch_size=256
    )

    analyzer = EmotionNeuralAnalyzer(config)

    # 构建词表
    sample_texts = ["今天很开心", "我很难过", "生气", "害怕"]
    analyzer.build_tokenizer(sample_texts)

    # 构建模型
    analyzer.build_model()

    # 模拟数据测试
    batch_size = 32
    seq_len = 64

    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    outputs = analyzer.model(dummy_input)

    print(f"\n输出形状:")
    print(f"  emotion_logits: {outputs['emotion_logits'].shape}")
    print(f"  vad_pred: {outputs['vad_pred'].shape}")
    print(f"  intensity_pred: {outputs['intensity_pred'].shape}")
    print(f"  irony_pred: {outputs['irony_pred'].shape}")

    print("\n模型测试通过!")
