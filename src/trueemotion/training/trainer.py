# -*- coding: utf-8 -*-
"""
TrueEmotion Trainer - 神经网络多任务训练器
==========================================

训练神经网络情感分析模型
支持多任务学习：情感分类 + VAD回归 + 强度回归 + 反讽检测
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trueemotion.models.neural_model import (
    EmotionNeuralAnalyzer, NeuralModelConfig, EmotionDataset
)
from trueemotion.data.data_generator import EmotionDataGenerator
from trueemotion.data.data_generator import EmotionDataGenerator, DistributedDataGenerator


# ==================== 训练配置 ====================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    train_samples: int = 1_000_000  # 100万训练样本
    val_samples: int = 10_000        # 1万验证样本
    test_samples: int = 10_000       # 1万测试样本
    irony_ratio: float = 0.15        # 15%反讽样本

    # 模型
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    # 训练
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # 损失权重
    emotion_weight: float = 1.0
    vad_weight: float = 0.3
    intensity_weight: float = 0.5
    irony_weight: float = 0.8

    # 其他
    save_interval: int = 5          # 每N轮保存
    print_interval: int = 100         # 每N步打印
    num_workers: int = 4             # 数据生成进程数


# ==================== 训练器 ====================

class EmotionTrainer:
    """
    情感模型训练器

    支持：
    - 自动数据生成
    - 多任务训练
    - 验证集评估
    - 断点续训
    - 训练可视化
    """

    def __init__(self, config: TrainingConfig, model_dir: str = None):
        self.config = config
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 模型
        self.analyzer = EmotionNeuralAnalyzer(self._build_model_config())
        self.trained_epochs = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

    def _build_model_config(self) -> NeuralModelConfig:
        """从训练配置构建模型配置"""
        return NeuralModelConfig(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            epochs=self.config.epochs,
            emotion_weight=self.config.emotion_weight,
            vad_weight=self.config.vad_weight,
            intensity_weight=self.config.intensity_weight,
            irony_weight=self.config.irony_weight
        )

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备数据"""
        print("\n" + "=" * 60)
        print("生成训练数据...")
        print("=" * 60)

        total_samples = self.config.train_samples + self.config.val_samples + self.config.test_samples

        # 使用单进程数据生成（更稳定）
        generator = EmotionDataGenerator(seed=42)
        texts, emotion_labels, vad_labels, intensity_labels, irony_labels = generator.generate(
            total_samples=total_samples,
            irony_ratio=self.config.irony_ratio
        )

        # 构建tokenizer
        self.analyzer.build_tokenizer(texts)

        # 创建完整数据集
        labels = {
            "emotion": emotion_labels,
            "vad": vad_labels,
            "intensity": intensity_labels,
            "irony": irony_labels
        }
        full_dataset = EmotionDataset(texts, labels, self.analyzer.tokenizer)

        # 划分数据集
        total_len = len(full_dataset)
        train_size = self.config.train_samples
        val_size = self.config.val_samples
        test_size = self.config.test_samples

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset):,} 样本")
        print(f"  验证集: {len(val_dataset):,} 样本")
        print(f"  测试集: {len(test_dataset):,} 样本")

        return train_loader, val_loader, test_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              resume: bool = False, checkpoint_path: str = None) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            resume: 是否从检查点恢复
            checkpoint_path: 检查点路径
        """
        # 构建模型
        self.analyzer.build_model()

        # 恢复检查点
        if resume and checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # 优化器
        optimizer = AdamW(
            self.analyzer.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        print("\n" + "=" * 60)
        print(f"开始训练 ({self.config.epochs} 轮)")
        print("=" * 60)

        for epoch in range(self.trained_epochs, self.config.epochs):
            epoch_start = time.time()

            # 训练
            train_metrics = self._train_epoch(epoch, train_loader, optimizer, scheduler)

            # 验证
            val_metrics = self._evaluate(val_loader)

            epoch_time = time.time() - epoch_start

            # 记录历史
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)

            # 打印
            self._print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)

            # 保存检查点
            self.trained_epochs = epoch + 1
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)

            # 保存最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_best_model()

        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss
        }

    def _train_epoch(self, epoch: int, dataloader: DataLoader,
                    optimizer, scheduler) -> Dict[str, float]:
        """训练单轮"""
        self.analyzer.model.train()

        total_loss = 0
        emotion_loss = 0
        vad_loss = 0
        intensity_loss = 0
        irony_loss = 0
        num_batches = 0

        criterion = torch.nn.BCEWithLogitsLoss()
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            targets = {
                "emotion_labels": batch["emotion_labels"].to(self.device),
                "vad_labels": batch["vad_labels"].to(self.device),
                "intensity_labels": batch["intensity_labels"].to(self.device),
                "irony_labels": batch["irony_labels"].to(self.device)
            }

            optimizer.zero_grad()
            outputs = self.analyzer.model(input_ids)

            # 多任务损失
            e_loss = criterion(outputs["emotion_logits"], targets["emotion_labels"])
            v_loss = mse_loss(outputs["vad_pred"], targets["vad_labels"])
            i_loss = mse_loss(outputs["intensity_pred"], targets["intensity_labels"])
            ir_loss = bce_loss(outputs["irony_pred"], targets["irony_labels"])

            loss = (
                self.config.emotion_weight * e_loss +
                self.config.vad_weight * v_loss +
                self.config.intensity_weight * i_loss +
                self.config.irony_weight * ir_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.analyzer.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            emotion_loss += e_loss.item()
            vad_loss += v_loss.item()
            intensity_loss += i_loss.item()
            irony_loss += ir_loss.item()
            num_batches += 1

            # 打印进度
            if (step + 1) % self.config.print_interval == 0:
                avg_loss = total_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                progress = (step + 1) / len(dataloader) * 100
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | {progress:.1f}%")

        return {
            "loss": total_loss / num_batches,
            "emotion_loss": emotion_loss / num_batches,
            "vad_loss": vad_loss / num_batches,
            "intensity_loss": intensity_loss / num_batches,
            "irony_loss": irony_loss / num_batches
        }

    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.analyzer.model.eval()

        total_loss = 0
        emotion_correct = 0
        vad_mae = 0
        intensity_mae = 0
        irony_correct = 0
        total_samples = 0

        primary_indices = list(range(8))

        criterion = torch.nn.BCEWithLogitsLoss()
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                targets = {
                    "emotion_labels": batch["emotion_labels"].to(self.device),
                    "vad_labels": batch["vad_labels"].to(self.device),
                    "intensity_labels": batch["intensity_labels"].to(self.device),
                    "irony_labels": batch["irony_labels"].to(self.device)
                }

                outputs = self.analyzer.model(input_ids)

                # 损失
                e_loss = criterion(outputs["emotion_logits"], targets["emotion_labels"])
                v_loss = mse_loss(outputs["vad_pred"], targets["vad_labels"])
                i_loss = mse_loss(outputs["intensity_pred"], targets["intensity_labels"])
                ir_loss = bce_loss(outputs["irony_pred"], targets["irony_labels"])

                loss = (
                    self.config.emotion_weight * e_loss +
                    self.config.vad_weight * v_loss +
                    self.config.intensity_weight * i_loss +
                    self.config.irony_weight * ir_loss
                )
                total_loss += loss.item()

                # 情感准确率
                emotion_probs = torch.sigmoid(outputs["emotion_logits"])
                pred_primary = emotion_probs[:, primary_indices].argmax(dim=1)
                true_primary = batch["emotion_labels"][:, primary_indices].argmax(dim=1).to(self.device)
                emotion_correct += (pred_primary == true_primary).sum().item()

                # VAD MAE
                vad_mae += torch.abs(outputs["vad_pred"] - targets["vad_labels"]).mean().item() * 3

                # 强度 MAE
                intensity_mae += torch.abs(outputs["intensity_pred"] - targets["intensity_labels"]).mean().item()

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

    def _print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float],
                              val_metrics: Dict[str, float], epoch_time: float):
        """打印轮次摘要"""
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"  训练损失: {train_metrics['loss']:.4f} | "
              f"情感损失: {train_metrics['emotion_loss']:.4f} | "
              f"VAD损失: {train_metrics['vad_loss']:.4f}")
        print(f"  验证损失: {val_metrics['loss']:.4f} | "
              f"情感准确率: {val_metrics['emotion_acc']:.2f}% | "
              f"VAD MAE: {val_metrics['vad_mae']:.4f}")
        print(f"  强度MAE: {val_metrics['intensity_mae']:.4f} | "
              f"反讽准确率: {val_metrics['irony_acc']:.2f}% | "
              f"用时: {epoch_time:.1f}s")

    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.analyzer.model.state_dict(),
            "optimizer_state": None,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss
        }, checkpoint_path)
        print(f"  检查点已保存: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.analyzer.model.load_state_dict(checkpoint["model_state"])
        self.trained_epochs = checkpoint["epoch"] + 1
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  已恢复检查点: epoch {self.trained_epochs}")

    def _save_best_model(self):
        """保存最佳模型"""
        best_path = self.model_dir / "best_model.pt"
        self.analyzer.save(str(best_path))
        print(f"  最佳模型已更新: {best_path}")

    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        """在测试集上评估"""
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        # 加载最佳模型
        best_path = self.model_dir / "best_model.pt"
        if best_path.exists():
            self.analyzer.load(str(best_path))

        test_metrics = self._evaluate(test_loader)

        print(f"\n测试结果:")
        print(f"  情感准确率: {test_metrics['emotion_acc']:.2f}%")
        print(f"  VAD MAE: {test_metrics['vad_mae']:.4f}")
        print(f"  强度 MAE: {test_metrics['intensity_mae']:.4f}")
        print(f"  反讽准确率: {test_metrics['irony_acc']:.2f}%")

        return test_metrics


def quick_train(num_samples: int = 100_000, epochs: int = 10):
    """
    快速训练（用于测试）

    Args:
        num_samples: 样本数量
        epochs: 训练轮数
    """
    config = TrainingConfig(
        train_samples=num_samples,
        val_samples=num_samples // 10,
        test_samples=num_samples // 10,
        batch_size=128,
        epochs=epochs,
        print_interval=50,
        save_interval=epochs + 1
    )

    trainer = EmotionTrainer(config)
    train_loader, val_loader, test_loader = trainer.prepare_data()
    trainer.train(train_loader, val_loader)
    trainer.evaluate_test(test_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrueEmotion 训练器")
    parser.add_argument("--samples", type=int, default=100_000, help="训练样本数")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch", type=int, default=256, help="批次大小")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TrueEmotion 快速训练")
    print(f"样本数: {args.samples:,} | 轮数: {args.epochs} | 批次: {args.batch}")
    print("=" * 70)

    quick_train(num_samples=args.samples, epochs=args.epochs)
