# -*- coding: utf-8 -*-
"""
TrueEmotion v4 Model - 单标签分类版本
=====================================

改进点：
1. 单标签分类（26选1 softmax）- 比多标签更容易训练
2. 去掉irony独立头，改为特征
3. 更高效的BiGRU + Attention
"""

import os
import sys
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


@dataclass
class V4Config:
    embedding_dim: int = 128
    hidden_dim: int = 192
    num_classes: int = 26
    dropout: float = 0.2
    max_words: int = 32
    patience: int = 8


class JiebaTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def fit(self, texts: List[str], max_vocab: int = 15000):
        word_counts = {}
        for text in texts:
            for w in jieba.lcut(text):
                word_counts[w] = word_counts.get(w, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for i, (word, _) in enumerate(sorted_words[:max_vocab - 2], start=2):
            self.word2idx[word] = i
            self.idx2word[i] = word
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str, max_len: int = 32) -> np.ndarray:
        words = jieba.lcut(text)
        result = [self.word2idx.get(w, 1) for w in words[:max_len]]
        while len(result) < max_len:
            result.append(0)
        return np.array(result, dtype=np.int64)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word, "vocab_size": self.vocab_size}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
            self.vocab_size = data["vocab_size"]


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class EmotionV4Model(nn.Module):
    """单标签分类的情感模型"""

    def __init__(self, config: V4Config, vocab_size: int):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim, num_layers=1,
                         batch_first=True, bidirectional=True, dropout=0)
        self.attention = Attention(config.hidden_dim * 2)

        hidden_dim = config.hidden_dim * 2

        # 情感分类 - 单标签 softmax
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.num_classes)
        )

        # VAD回归
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # 强度回归
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        context, _ = self.attention(gru_out)

        emotion_logits = self.emotion_head(context)  # (batch, 26)
        vad_pred = torch.tanh(self.vad_head(context))  # (batch, 3)
        intensity_pred = self.intensity_head(context)  # (batch, 1)

        return {
            "emotion_logits": emotion_logits,
            "vad_pred": vad_pred,
            "intensity_pred": intensity_pred.squeeze(-1)
        }


class EmotionV4Dataset(Dataset):
    def __init__(self, texts, emotion_idxs, vad_labels, intensity_labels, tokenizer, max_len=32):
        self.texts = texts
        self.emotion_idxs = emotion_idxs
        self.vad_labels = vad_labels
        self.intensity_labels = intensity_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.texts[idx], self.max_len)
        return {
            "input_ids": torch.tensor(enc, dtype=torch.long),
            "emotion_idx": torch.tensor(self.emotion_idxs[idx], dtype=torch.long),
            "vad_labels": torch.tensor(self.vad_labels[idx], dtype=torch.float),
            "intensity_labels": torch.tensor(self.intensity_labels[idx], dtype=torch.float)
        }


class EmotionV4Analyzer:
    """v4分析器"""

    EMOTION_NAMES = [
        "joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust",
        "optimism", "love", "guilt", "submission", "surprise_complex", "disappointment",
        "remorse", "envy", "suspicion", "aggression", "pride", "contentment", "contempt",
        "cynicism", "morbidness", "sentimentality", "anxiety", "despair"
    ]

    def __init__(self, config=None):
        self.config = config or V4Config()
        self.tokenizer = JiebaTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_tokenizer(self, texts):
        self.tokenizer.fit(texts)
        print(f"词表大小: {self.tokenizer.vocab_size}")

    def build_model(self):
        self.model = EmotionV4Model(self.config, self.tokenizer.vocab_size)
        self.model.to(self.device)
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_data(self, texts, emotion_idxs, vad_labels, intensity_labels, batch_size=128):
        dataset = EmotionV4Dataset(texts, emotion_idxs, vad_labels, intensity_labels, self.tokenizer, self.config.max_words)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def train(self, train_loader, val_loader, epochs=30, lr=1e-3, save_path="models/v4_model.pt"):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        best_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                emotion_idx = batch["emotion_idx"].to(self.device)
                vad_labels = batch["vad_labels"].to(self.device)
                intensity_labels = batch["intensity_labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids)

                # 单标签交叉熵损失
                ce_loss = F.cross_entropy(outputs["emotion_logits"], emotion_idx)
                vad_loss = F.mse_loss(outputs["vad_pred"], vad_labels)
                intensity_loss = F.mse_loss(outputs["intensity_pred"], intensity_labels)

                loss = ce_loss + 0.3 * vad_loss + 0.5 * intensity_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            # 验证
            val_metrics = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.2f}%")

            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                self.save(save_path)
                print(f"  -> Best model saved (acc: {best_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return {"best_acc": best_acc}

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            emotion_idx = batch["emotion_idx"].to(self.device)
            vad_labels = batch["vad_labels"].to(self.device)
            intensity_labels = batch["intensity_labels"].to(self.device)

            outputs = self.model(input_ids)

            ce_loss = F.cross_entropy(outputs["emotion_logits"], emotion_idx)
            vad_loss = F.mse_loss(outputs["vad_pred"], vad_labels)
            intensity_loss = F.mse_loss(outputs["intensity_pred"], intensity_labels)
            loss = ce_loss + 0.3 * vad_loss + 0.5 * intensity_loss

            total_loss += loss.item()

            pred = outputs["emotion_logits"].argmax(dim=1)
            correct += (pred == emotion_idx).sum().item()
            total += input_ids.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "acc": correct / total * 100
        }

    @torch.no_grad()
    def predict(self, texts, batch_size=64):
        self.model.eval()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = [self.tokenizer.encode(t, self.config.max_words) for t in batch_texts]
            input_ids = torch.tensor(encodings, dtype=torch.long).to(self.device)

            outputs = self.model(input_ids)

            probs = F.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()
            vad_preds = outputs["vad_pred"].cpu().numpy()
            intensity_preds = outputs["intensity_pred"].cpu().numpy()

            for j in range(len(batch_texts)):
                idx = probs[j].argmax()
                results.append({
                    "text": batch_texts[j],
                    "primary_emotion": self.EMOTION_NAMES[idx],
                    "primary_score": float(probs[j, idx]),
                    "vad": tuple(float(v) for v in vad_preds[j]),
                    "intensity": float(intensity_preds[j])
                })

        return results

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "tokenizer": self.tokenizer
        }, path)
        print(f"Model saved: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.tokenizer = checkpoint["tokenizer"]
        self.build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        print(f"Model loaded: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("TrueEmotion v4 Model - 单标签分类版本")
    print("=" * 60)

    config = V4Config()
    analyzer = EmotionV4Analyzer(config)

    # 测试
    test_texts = ["今天很开心", "我很难过", "生气"]
    analyzer.build_tokenizer(test_texts)
    analyzer.build_model()

    # 测试预测
    dummy_texts = ["项目完成了"]
    results = analyzer.predict(dummy_texts)
    print(f"\n预测: {results}")
