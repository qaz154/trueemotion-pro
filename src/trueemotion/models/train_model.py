# -*- coding: utf-8 -*-
"""
TrueEmotion Model Training Script
训练脚本 - 训练字符级CNN情感模型
"""

import os
import sys
import json
import random
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 字符级模型定义 ====================

class CharConfig:
    """模型配置"""
    embedding_dim: int = 128
    hidden_dim: int = 192
    num_classes: int = 15  # 15种情感类别
    dropout: float = 0.25
    max_len: int = 64


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
        with open(path, 'wb') as f:
            pickle.dump({
                "char2idx": self.char2idx,
                "idx2char": self.idx2char,
                "vocab_size": self.vocab_size
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char2idx = data["char2idx"]
            self.idx2char = data["idx2char"]
            self.vocab_size = data["vocab_size"]


class CharCNN(nn.Module):
    """字符级CNN情感分类器"""
    def __init__(self, config: CharConfig, vocab_size: int):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.hidden_dim, kernel_size=k)
            for k in [2, 3, 4]
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 3, config.num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)  # (batch, hidden_dim * 3)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==================== 数据集 ====================

EMOTION_LABELS = [
    'joy', 'sadness', 'anger', 'fear', 'surprise',
    'anticipation', 'trust', 'disgust', 'optimism', 'love',
    'guilt', 'envy', 'despair', 'anxiety', 'neutral'
]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTION_LABELS)}


class EmotionDataset(Dataset):
    """情感数据集"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: CharacterTokenizer, max_len: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode(text, self.max_len)
        return torch.tensor(encoding, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ==================== 训练数据生成 ====================

def generate_training_data(n_samples: int = 5000) -> Tuple[List[str], List[int]]:
    """生成训练数据"""
    templates = {
        'joy': ['太开心了', '真高兴', '太棒了', '好开心', '真棒', '太美了', '好高兴啊'],
        'sadness': ['好难过', '心里难受', '好伤心', '难过了', '失落', '沮丧', '郁闷'],
        'anger': ['气死了', '太气人了', '真生气', '可恶', '讨厌', '烦死了', '火大'],
        'fear': ['好害怕', '担心', '紧张', '害怕', '不安', '惶恐', '担忧'],
        'surprise': ['太意外了', '没想到', '震惊', '吃惊', '惊讶', '哇', '什么'],
        'anticipation': ['期待', '希望', '盼望', '憧憬', '希望', '期待中', '希望'],
        'trust': ['相信', '信任', '放心', '依赖', '托付', '信赖', '相信'],
        'disgust': ['恶心', '讨厌', '厌恶', '反感', '嫌弃', '不屑', '鄙视'],
        'optimism': ['会好的', '曙光', '加油', '积极', '乐观', '阳光', '有信心'],
        'love': ['喜欢', '爱', '心动', '甜蜜', '浪漫', '温馨', '温柔'],
        'guilt': ['愧疚', '自责', '抱歉', '对不起', '过意不去', '后悔', '遗憾'],
        'envy': ['羡慕', '嫉妒', '眼红', '不平衡', '柠檬', '酸', '眼馋'],
        'despair': ['绝望', '没希望', '放弃', '崩溃', '死心', '无望', '彻底绝望'],
        'anxiety': ['焦虑', '着急', '忧虑', '忐忑', '心慌', '发愁', '压力'],
        'neutral': ['好的', '知道了', '嗯', '好吧', '随便', '无所谓', '一般'],
    }

    texts, labels = [], []
    for _ in range(n_samples):
        emotion = random.choice(EMOTION_LABELS)
        template = random.choice(templates[emotion])
        # 添加噪声
        suffixes = ['', '', '', '啊', '吧', '呢', '。', '！', '...']
        prefix = random.choice(['', '今天', '现在', '我', '感觉', '真的'])
        text = prefix + template + random.choice(suffixes)
        texts.append(text)
        labels.append(EMOTION_TO_IDX[emotion])

    return texts, labels


def augment_text(text: str) -> str:
    """数据增强 - 添加同义词替换"""
    # 简单增强：添加一些随机字符
    if random.random() < 0.3:
        noise_chars = '呀嘛哈嗯哦呃'
        pos = random.randint(0, len(text))
        text = text[:pos] + random.choice(noise_chars) + text[pos:]
    return text


# ==================== 训练函数 ====================

def train_model(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    model_save_path: str = None,
    data_save_path: str = None
):
    """训练模型"""
    print("=" * 60)
    print("TrueEmotion 字符级CNN模型训练")
    print("=" * 60)

    # 生成数据
    print("\n[1] 生成训练数据...")
    texts, labels = generate_training_data(n_samples=5000)

    # 数据增强
    texts = [augment_text(t) for t in texts]

    # 划分训练集和验证集
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # 创建tokenizer
    print("[2] 构建字符表...")
    tokenizer = CharacterTokenizer()
    tokenizer.fit(train_texts + val_texts)
    print(f"    字符表大小: {tokenizer.vocab_size}")

    # 创建数据集
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 创建模型
    print("[3] 创建模型...")
    config = CharConfig()
    model = CharCNN(config, tokenizer.vocab_size)
    print(f"    模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练配置
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("[4] 开始训练...")
    best_val_acc = 0
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()

        scheduler.step()

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_correct += (outputs.argmax(1) == batch_y).sum().item()

        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)

        print(f"    Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss/len(train_loader):.4f} - "
              f"Train: {train_acc:.4f} - "
              f"Val: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")

    # 保存模型
    if model_save_path:
        print(f"\n[5] 保存模型到 {model_save_path}...")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': tokenizer.vocab_size
        }, model_save_path)

        # 保存tokenizer
        tokenizer_path = model_save_path.replace('.pt', '_tokenizer.pkl')
        tokenizer.save(tokenizer_path)
        print(f"    Tokenizer保存到 {tokenizer_path}")

    # 保存数据
    if data_save_path:
        print(f"\n[6] 保存数据到 {data_save_path}...")
        os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
        data = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'emotion_labels': EMOTION_LABELS
        }
        with open(data_save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n训练完成!")
    return model, tokenizer


# ==================== 预测函数 ====================

def load_trained_model(model_path: str, tokenizer_path: str):
    """加载训练好的模型"""
    # 加载tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.load(tokenizer_path)

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    model = CharCNN(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, text: str, device: str = 'cpu') -> Dict:
    """预测单个文本"""
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode(text, 64)
        x = torch.tensor(encoding, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(1).item()
        pred_emotion = IDX_TO_EMOTION[pred_idx]
        confidence = probs[0][pred_idx].item()

        return {
            'emotion': pred_emotion,
            'confidence': confidence,
            'all_probs': {EMOTION_LABELS[i]: probs[0][i].item() for i in range(len(EMOTION_LABELS))}
        }


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练TrueEmotion模型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--model_path', type=str, default='models/char_emotion_model.pt', help='模型保存路径')
    parser.add_argument('--data_path', type=str, default='models/training_data.json', help='数据保存路径')

    args = parser.parse_args()

    # 训练
    model, tokenizer = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_path=args.model_path,
        data_save_path=args.data_path
    )

    # 测试预测
    print("\n测试预测:")
    test_texts = [
        "太开心了！",
        "工作好累啊",
        "气死了！",
        "被裁员了..."
    ]
    for text in test_texts:
        result = predict(model, tokenizer, text)
        print(f"  {text} -> {result['emotion']} ({result['confidence']:.3f})")
