# -*- coding: utf-8 -*-
"""
TrueEmotion Optimized Neural Model - 优化版神经网络模型
=====================================================

优化点：
1. jieba分词 - 更精准的中文处理
2. 可训练词向量 - 比字符级更高效
3. 单层BiGRU - 比LSTM更快
4. 早停机制 - 防过拟合
5. 梯度累积 - 模拟大batch

架构：Embedding → BiGRU → Attention → Multi-task Heads
"""

import os
import sys
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
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jieba


# ==================== 配置 ====================

@dataclass
class OptimizedModelConfig:
    """优化版模型配置"""
    # 词向量维度
    embedding_dim: int = 100
    # 隐藏层维度
    hidden_dim: int = 128
    # Dropout概率
    dropout: float = 0.2
    # 情感类别数
    num_classes: int = 26
    # 早停参数
    patience: int = 5
    # 梯度累积步数
    grad_accum_steps: int = 4
    # 最大词数
    max_words: int = 32


# ==================== 分词器 ====================

class JiebaTokenizer:
    """基于jieba的分词器"""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def fit(self, texts: List[str], max_vocab: int = 15000):
        """从文本学习词表"""
        word_counts = {}
        for text in texts:
            words = jieba.lcut(text)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # 按频率排序，保留top词汇
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for i, (word, _) in enumerate(sorted_words[:max_vocab - 2], start=2):
            self.word2idx[word] = i
            self.idx2word[i] = word
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str, max_len: int = 32) -> np.ndarray:
        """编码文本为索引序列"""
        words = jieba.lcut(text)
        result = []
        for w in words[:max_len]:
            result.append(self.word2idx.get(w, 1))
        while len(result) < max_len:
            result.append(0)
        return np.array(result, dtype=np.int64)

    def decode(self, indices: np.ndarray) -> str:
        """解码回文本"""
        return "".join(self.idx2word.get(i, "<UNK>") for i in indices)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word, "vocab_size": self.vocab_size}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
            self.vocab_size = data["vocab_size"]


# ==================== 注意力层 ====================

class Attention(nn.Module):
    """简单注意力层"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None):
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        if mask is not None:
            attention_weights = attention_weights * mask.unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


# ==================== 优化模型 ====================

class OptimizedEmotionModel(nn.Module):
    """
    优化版情感分析模型

    使用BiGRU + Attention，比LSTM更快
    """

    def __init__(self, config: OptimizedModelConfig, vocab_size: int):
        super().__init__()
        self.config = config

        # 词向量层
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)

        # 单层BiGRU（比LSTM快）
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # 注意力
        self.attention = Attention(config.hidden_dim * 2)

        # 输出层
        hidden_dim = config.hidden_dim * 2

        # 情感分类 (多标签)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, config.num_classes)
        )

        # VAD回归
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

        # 强度回归
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 反讽检测
        self.irony_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, seq_len)

        # 词向量
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # BiGRU
        gru_output, _ = self.gru(embedded)  # (batch, seq_len, hidden_dim * 2)

        # 注意力
        context, _ = self.attention(gru_output)  # (batch, hidden_dim * 2)

        # 多任务输出
        emotion_logits = self.emotion_head(context)
        vad_pred = torch.tanh(self.vad_head(context))
        intensity_pred = self.intensity_head(context)
        irony_pred = self.irony_head(context)

        return {
            "emotion_logits": emotion_logits,
            "vad_pred": vad_pred,
            "intensity_pred": intensity_pred,
            "irony_pred": irony_pred
        }


# ==================== 数据集 ====================

class OptimizedEmotionDataset(Dataset):
    """优化版数据集"""

    def __init__(self, texts: List[str], emotion_labels: np.ndarray,
                 vad_labels: np.ndarray, intensity_labels: np.ndarray,
                 irony_labels: np.ndarray, tokenizer: JiebaTokenizer, max_len: int = 32):
        self.texts = texts
        self.emotion_labels = emotion_labels
        self.vad_labels = vad_labels
        self.intensity_labels = intensity_labels
        self.irony_labels = irony_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx], self.max_len)
        return {
            "input_ids": torch.tensor(encoding, dtype=torch.long),
            "emotion_labels": torch.tensor(self.emotion_labels[idx], dtype=torch.float),
            "vad_labels": torch.tensor(self.vad_labels[idx], dtype=torch.float),
            "intensity_labels": torch.tensor(self.intensity_labels[idx], dtype=torch.float),
            "irony_labels": torch.tensor(self.irony_labels[idx], dtype=torch.float)
        }


# ==================== 早停追踪器 ====================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ==================== 优化版分析器 ====================

class OptimizedEmotionAnalyzer:
    """优化版情感分析器"""

    EMOTION_NAMES = [
        "joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust",
        "optimism", "love", "guilt", "submission", "surprise_complex", "disappointment",
        "remorse", "envy", "suspicion", "aggression", "pride", "contentment", "contempt",
        "cynicism", "morbidness", "sentimentality", "anxiety", "despair"
    ]

    def __init__(self, config: Optional[OptimizedModelConfig] = None):
        self.config = config or OptimizedModelConfig()
        self.tokenizer = JiebaTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping = EarlyStopping(patience=self.config.patience)

    def build_tokenizer(self, texts: List[str]):
        """构建分词器"""
        self.tokenizer.fit(texts, max_vocab=15000)
        print(f"词表大小: {self.tokenizer.vocab_size}")

    def build_model(self):
        """构建模型"""
        self.model = OptimizedEmotionModel(self.config, self.tokenizer.vocab_size)
        self.model.to(self.device)
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_data(self, texts: List[str], emotion_labels: np.ndarray,
                    vad_labels: np.ndarray, intensity_labels: np.ndarray,
                    irony_labels: np.ndarray, batch_size: int = 128) -> DataLoader:
        """准备数据加载器"""
        dataset = OptimizedEmotionDataset(
            texts, emotion_labels, vad_labels, intensity_labels, irony_labels,
            self.tokenizer, max_len=self.config.max_words
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 30, lr: float = 1e-3, save_path: str = "models/optimized_model.pt") -> Dict[str, Any]:
        """训练模型"""
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        history = []
        best_val_loss = float('inf')
        grad_accum_steps = self.config.grad_accum_steps

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                targets = {
                    "emotion_labels": batch["emotion_labels"].to(self.device),
                    "vad_labels": batch["vad_labels"].to(self.device),
                    "intensity_labels": batch["intensity_labels"].to(self.device),
                    "irony_labels": batch["irony_labels"].to(self.device)
                }

                outputs = self.model(input_ids)

                # 多任务损失
                e_loss = criterion(outputs["emotion_logits"], targets["emotion_labels"])
                v_loss = mse_loss(outputs["vad_pred"], targets["vad_labels"])
                i_loss = mse_loss(outputs["intensity_pred"], targets["intensity_labels"])
                ir_loss = bce_loss(outputs["irony_pred"], targets["irony_labels"])

                loss = e_loss + 0.3 * v_loss + 0.5 * i_loss + 0.8 * ir_loss
                loss = loss / grad_accum_steps
                loss.backward()

                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * grad_accum_steps

            scheduler.step()

            # 验证
            val_metrics = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_metrics["loss"],
                "emotion_acc": val_metrics["emotion_acc"],
                "vad_mae": val_metrics["vad_mae"],
                "irony_acc": val_metrics["irony_acc"]
            })

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                  f"Emotion Acc: {val_metrics['emotion_acc']:.2f}% | "
                  f"VAD MAE: {val_metrics['vad_mae']:.4f}")

            # 保存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save(save_path)
                print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")

            # 早停检查
            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch+1}")
                break

        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        total_loss = 0
        emotion_correct = 0
        vad_mae = 0
        irony_correct = 0
        total_samples = 0

        criterion = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = {
                "emotion_labels": batch["emotion_labels"].to(self.device),
                "vad_labels": batch["vad_labels"].to(self.device),
                "intensity_labels": batch["intensity_labels"].to(self.device),
                "irony_labels": batch["irony_labels"].to(self.device)
            }

            outputs = self.model(input_ids)

            e_loss = criterion(outputs["emotion_logits"], targets["emotion_labels"])
            v_loss = mse_loss(outputs["vad_pred"], targets["vad_labels"])
            i_loss = mse_loss(outputs["intensity_pred"], targets["intensity_labels"])
            ir_loss = bce_loss(outputs["irony_pred"], targets["irony_labels"])

            loss = e_loss + 0.3 * v_loss + 0.5 * i_loss + 0.8 * ir_loss
            total_loss += loss.item()

            # 情感准确率
            emotion_probs = torch.sigmoid(outputs["emotion_logits"])
            pred_primary = emotion_probs[:, :8].argmax(dim=1)
            true_primary = batch["emotion_labels"][:, :8].argmax(dim=1).to(self.device)
            emotion_correct += (pred_primary == true_primary).sum().item()

            # VAD MAE
            vad_mae += torch.abs(outputs["vad_pred"] - targets["vad_labels"]).mean().item() * 3

            # 反讽准确率
            irony_pred = (outputs["irony_pred"] > 0.5).float().squeeze(-1)
            irony_correct += (irony_pred == targets["irony_labels"].squeeze(-1)).sum().item()

            total_samples += input_ids.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "emotion_acc": emotion_correct / total_samples * 100,
            "vad_mae": vad_mae / len(dataloader),
            "irony_acc": irony_correct / total_samples * 100
        }

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """预测"""
        self.model.eval()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = [self.tokenizer.encode(t, self.config.max_words) for t in batch_texts]
            input_ids = torch.tensor(encodings, dtype=torch.long).to(self.device)

            outputs = self.model(input_ids)

            emotion_probs = torch.sigmoid(outputs["emotion_logits"]).cpu().numpy()
            vad_preds = outputs["vad_pred"].cpu().numpy()
            intensity_preds = outputs["intensity_pred"].cpu().numpy()
            irony_preds = outputs["irony_pred"].cpu().numpy()

            for j in range(len(batch_texts)):
                primary_scores = emotion_probs[j, :8]
                primary_idx = primary_scores.argmax()
                primary_emotion = self.EMOTION_NAMES[primary_idx]

                complex_emotions = []
                for k in range(8, 26):
                    if emotion_probs[j, k] > 0.4:
                        complex_emotions.append(self.EMOTION_NAMES[k])

                results.append({
                    "text": batch_texts[j],
                    "primary_emotion": primary_emotion,
                    "primary_score": float(primary_scores[primary_idx]),
                    "complex_emotions": complex_emotions,
                    "vad": tuple(float(v) for v in vad_preds[j]),
                    "intensity": float(intensity_preds[j, 0]),
                    "is_irony": bool(irony_preds[j, 0] > 0.5),
                    "irony_prob": float(irony_preds[j, 0])
                })

        return results

    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    print("=" * 60)
    print("TrueEmotion Optimized Model - 优化版神经网络")
    print("=" * 60)

    config = OptimizedModelConfig()
    analyzer = OptimizedEmotionAnalyzer(config)

    # 测试分词
    test_texts = ["今天很开心", "我很难过", "项目终于完成了"]
    analyzer.build_tokenizer(test_texts)

    for text in test_texts:
        words = jieba.lcut(text)
        print(f"  {text} -> {words}")

    # 测试模型
    analyzer.build_model()

    # 模拟输入
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    outputs = analyzer.model(dummy_input)

    print(f"\n输出形状:")
    print(f"  emotion_logits: {outputs['emotion_logits'].shape}")
    print(f"  vad_pred: {outputs['vad_pred'].shape}")
    print(f"  intensity_pred: {outputs['intensity_pred'].shape}")
    print(f"  irony_pred: {outputs['irony_pred'].shape}")

    print("\n模型测试通过!")
