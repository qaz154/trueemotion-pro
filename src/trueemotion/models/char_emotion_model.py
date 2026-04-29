# -*- coding: utf-8 -*-
"""
Character-level Emotion Model
字符级神经网络模型 - 更好地处理未登录词
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Optional
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


@dataclass
class CharConfig:
    embedding_dim: int = 128
    hidden_dim: int = 192
    num_classes: int = 26
    dropout: float = 0.25
    max_len: int = 64  # 字符级需要更长
    patience: int = 8


class CharacterTokenizer:
    """字符级分词器"""
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def fit(self, texts: List[str], max_vocab: int = 8000):
        """从文本学习字符表"""
        char_counts = {}
        for text in texts:
            for c in text:
                if '\u4e00' <= c <= '\u9fff' or c.isalnum():
                    char_counts[c] = char_counts.get(c, 0) + 1

        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        for i, (char, _) in enumerate(sorted_chars[:max_vocab - 2], start=2):
            self.char2idx[char] = i
            self.idx2char[i] = char
        self.vocab_size = len(self.char2idx)

    def encode(self, text: str, max_len: int = 64) -> np.ndarray:
        """编码文本为字符索引"""
        result = []
        for c in text[:max_len]:
            if c in self.char2idx:
                result.append(self.char2idx[c])
            elif '\u4e00' <= c <= '\u9fff':
                result.append(self.char2idx.get(c, 1))
            elif c.isalnum():
                result.append(self.char2idx.get(c, 1))
            else:
                continue
        while len(result) < max_len:
            result.append(0)
        return np.array(result, dtype=np.int64)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"char2idx": self.char2idx, "idx2char": self.idx2char, "vocab_size": self.vocab_size}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.char2idx = data["char2idx"]
            self.idx2char = data["idx2char"]
            self.vocab_size = data["vocab_size"]


class CharCNNModel(nn.Module):
    """字符级CNN模型"""
    def __init__(self, config: CharConfig, vocab_size: int):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)

        # 多尺度CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=k)
            for k in [2, 3, 4]
        ])

        self.dropout = nn.Dropout(config.dropout)

        # 分类头
        hidden_total = config.hidden_dim * 3
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_total, hidden_total // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_total // 2, config.num_classes)
        )

        self.vad_head = nn.Sequential(
            nn.Linear(hidden_total, hidden_total // 2),
            nn.ReLU(),
            nn.Linear(hidden_total // 2, 3)
        )

        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_total, hidden_total // 2),
            nn.ReLU(),
            nn.Linear(hidden_total // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # CNN需要 (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, hidden_dim, seq_len-ker+1)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)  # (batch, hidden_dim)
            conv_outputs.append(pooled)

        context = torch.cat(conv_outputs, dim=1)  # (batch, hidden_dim * 3)
        context = self.dropout(context)

        emotion_logits = self.emotion_head(context)
        vad_pred = torch.tanh(self.vad_head(context))
        intensity_pred = self.intensity_head(context)

        return {
            "emotion_logits": emotion_logits,
            "vad_pred": vad_pred,
            "intensity_pred": intensity_pred.squeeze(-1)
        }


class CharDataset(Dataset):
    def __init__(self, texts, emotion_idxs, vad_labels, intensity_labels, tokenizer, max_len=64):
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


class CharEmotionAnalyzer:
    """字符级情感分析器"""

    EMOTION_NAMES = [
        "joy", "trust", "fear", "anger", "surprise", "anticipation", "sadness", "disgust",
        "optimism", "love", "guilt", "submission", "surprise_complex", "disappointment",
        "remorse", "envy", "suspicion", "aggression", "pride", "contentment", "contempt",
        "cynicism", "morbidness", "sentimentality", "anxiety", "despair"
    ]

    def __init__(self, config: Optional[CharConfig] = None):
        self.config = config or CharConfig()
        self.tokenizer = CharacterTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_tokenizer(self, texts):
        self.tokenizer.fit(texts)
        print(f"Char vocab size: {self.tokenizer.vocab_size}")

    def build_model(self):
        self.model = CharCNNModel(self.config, self.tokenizer.vocab_size)
        self.model.to(self.device)
        print(f"Char model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_data(self, texts, emotion_idxs, vad_labels, intensity_labels, batch_size=128):
        dataset = CharDataset(texts, emotion_idxs, vad_labels, intensity_labels, self.tokenizer, self.config.max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def train(self, train_loader, val_loader, epochs=30, lr=1e-3, save_path="models/char_model.pt"):
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

                ce_loss = F.cross_entropy(outputs["emotion_logits"], emotion_idx)
                vad_loss = F.mse_loss(outputs["vad_pred"], vad_labels)
                intensity_loss = F.mse_loss(outputs["intensity_pred"], intensity_labels)

                loss = ce_loss + 0.3 * vad_loss + 0.5 * intensity_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            val_metrics = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.2f}%")

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
            encodings = [self.tokenizer.encode(t, self.config.max_len) for t in batch_texts]
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
    print("Character-level Emotion Model")
    print("=" * 60)

    config = CharConfig()
    analyzer = CharEmotionAnalyzer(config)

    test_texts = ["今天很开心", "我很难过", "项目终于完成了"]
    analyzer.build_tokenizer(test_texts)
    analyzer.build_model()

    results = analyzer.predict(test_texts)
    for r in results:
        print(f"{r['text']} -> {r['primary_emotion']} ({r['primary_score']:.3f})")