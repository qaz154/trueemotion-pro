# -*- coding: utf-8 -*-
"""
TrueEmotion 主入口
==================

新一代真实情感AI系统

功能：
1. 24种情感识别（8原始+16复合）
2. 情感强度预测（0-1连续值）
3. VAD维度连续表示
4. 反讽/隐喻检测
5. 上下文感知情感理解
6. 场景级泛化评估
"""

import argparse
import sys
from typing import List, Optional

from trueemotion.models.emotion_model import EmotionAnalyzer, TrueEmotionModel
from trueemotion.models.irony_detector import IronyDetector
from trueemotion.models.context_encoder import ContextEncoder
from trueemotion.emotion.plutchik24 import get_all_emotions, get_primary_emotions, get_complex_emotions
from trueemotion.data.scenario_generator import ScenarioBasedDataGenerator
from trueemotion.training.trainer import TrueEmotionTrainer, TrainingConfig


def print_banner():
    """打印横幅"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       🎭 TrueEmotion - 新一代真实情感AI 🎭                           ║
║                                                                      ║
║       24种情感 | 强度预测 | VAD维度 | 反讽检测 | 上下文理解          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def analyze_text(analyzer: EmotionAnalyzer, text: str, show_context: bool = False):
    """分析单条文本"""
    result = analyzer.analyze(text)

    print(f"\n📝 文本: {text}")
    print(f"   结果: {result}")

    # 详细信息
    print(f"\n   详细分析:")
    print(f"   ├─ 原型情感: {result.primary}")
    print(f"   ├─ 主要情感: {result.get_primary_emotion()}")
    print(f"   ├─ 复合情感: {result.get_complex_emotions()}")
    print(f"   ├─ VAD: V={result.vad[0]:.2f}, A={result.vad[1]:.2f}, D={result.vad[2]:.2f}")
    print(f"   ├─ 强度: {result.intensity:.2f}")
    print(f"   ├─ 置信度: {result.confidence:.2f}")
    print(f"   ├─ 反讽: {result.is_irony}", end="")
    if result.is_irony:
        print(f" (表面:{result.surface_emotion} → 真实:{result.true_emotion}, 置信度:{result.irony_confidence:.2f})")
    else:
        print()
    print(f"   └─ 状态: {result.state}")


def demo_analysis():
    """演示情感分析"""
    print("\n" + "=" * 70)
    print("🎭 情感分析演示")
    print("=" * 70)

    analyzer = EmotionAnalyzer()

    # 测试文本
    test_texts = [
        # 原始情感
        "今天太开心了！终于完成了项目！",
        "我很难过，失恋了...",
        "真是气死我了，又被骗了！",
        "好害怕啊，担心明天的考试...",
        "好期待！下个月就要去旅游了！",

        # 复合情感
        "看着他成功了，既高兴又有点嫉妒。",
        "又失望又生气，真是受够了！",

        # 反讽
        "还行吧，就那样。",
        "真是太感谢了，让我等了三个小时。",
        "太好了，又迟到了。",
        "好怕怕哦，吓死人了呢。",

        # 真正面
        "太开心了！终于成功了！",
        "谢谢你的礼物，我很喜欢！",

        # 中性
        "今天天气不错。",
        "嗯，我知道了。",
    ]

    for text in test_texts:
        analyze_text(analyzer, text)


def demo_context():
    """演示上下文理解"""
    print("\n" + "=" * 70)
    print("🔗 上下文理解演示")
    print("=" * 70)

    analyzer = EmotionAnalyzer()

    dialogues = [
        "我今天加班到很晚，好累啊。",
        "辛苦了，工作不要太拼。",
        "可是项目还是没完成...",
        "别着急，慢慢来。",
        "怎么办啊，真的很焦虑！"
    ]

    print("\n📖 对话历史:")
    for i, text in enumerate(dialogues, 1):
        result = analyzer.analyze(text)
        print(f"   {i}. [{result.get_primary_emotion():12s}] {text}")

    print("\n💡 注意: 系统会追踪对话历史中的情感变化")


def demo_scenarios():
    """演示场景数据生成"""
    print("\n" + "=" * 70)
    print("📊 场景数据演示")
    print("=" * 70)

    generator = ScenarioBasedDataGenerator()

    print("\n支持的情感场景:")
    for sid, scenario in generator.scenarios.items():
        emotions = ", ".join(scenario.emotions[:4])
        print(f"   • {scenario.name}: {emotions}...")

    print("\n示例样本:")
    samples = generator.generate_samples(6)
    for sample in samples[:3]:
        print(f"\n   [{sample['scenario_name']}]")
        print(f"   文本: {sample['text']}")
        print(f"   情感: {sample['primary_labels']}")
        print(f"   强度: {sample['intensity_label']:.2f}")


def run_training(train_samples: int = 2000, eval_samples: int = 500, epochs: int = 10):
    """运行训练"""
    print("\n" + "=" * 70)
    print("🧠 TrueEmotion 训练")
    print("=" * 70)

    config = TrainingConfig(
        train_samples=train_samples,
        eval_samples=eval_samples,
        epochs=epochs,
        unseen_scenarios=["health_concern", "financial_worry"]
    )

    trainer = TrueEmotionTrainer(config)
    report = trainer.train()

    return report


def run_full_test():
    """运行完整测试"""
    print("\n" + "=" * 70)
    print("🧪 TrueEmotion 完整测试")
    print("=" * 70)

    # 1. 情感分析测试
    print("\n【测试1】情感分析")
    analyzer = EmotionAnalyzer()

    test_cases = [
        ("太开心了！", "joy"),
        ("很难过...", "sadness"),
        ("气死我了！", "anger"),
        ("好害怕...", "fear"),
        ("好期待！", "anticipation"),
        ("惊呆了！", "surprise"),
        ("真恶心！", "disgust"),
        ("还行吧...", "neutral"),
    ]

    correct = 0
    for text, expected in test_cases:
        result = analyzer.analyze(text)
        pred = result.get_primary_emotion()
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"   {status} \"{text[:15]}...\" → 期望:{expected}, 预测:{pred}")

    print(f"\n   情感分类准确率: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")

    # 2. 反讽检测测试
    print("\n【测试2】反讽检测")
    irony_cases = [
        ("还行吧，就那样。", True),
        ("真是太感谢了，让我等了这么久。", True),
        ("太好了，又迟到了。", True),
        ("太开心了！终于成功了！", False),  # 真正面
    ]

    irony_correct = 0
    for text, expected in irony_cases:
        result = analyzer.analyze(text)
        pred = result.is_irony
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            irony_correct += 1
        print(f"   {status} \"{text[:20]}...\" → 期望:{'反讽' if expected else '正常'}, 预测:{'反讽' if pred else '正常'}")

    print(f"\n   反讽检测准确率: {irony_correct}/{len(irony_cases)} = {irony_correct/len(irony_cases):.1%}")

    # 3. 复合情感测试
    print("\n【测试3】复合情感识别")
    complex_cases = [
        "看着他成功了，既高兴又有点嫉妒。",
        "又失望又生气，真是受够了！",
        "又惊又喜，简直不敢相信！",
    ]

    for text in complex_cases:
        result = analyzer.analyze(text)
        complex_list = result.get_complex_emotions()
        print(f"   \"{text[:20]}...\"")
        print(f"      主要: {result.get_primary_emotion()}, 复合: {complex_list[:2] if complex_list else '无'}")

    # 4. VAD维度测试
    print("\n【测试4】VAD维度预测")
    vad_cases = [
        ("太开心了！", (0.9, 0.5, 0.7)),
        ("很难过...", (-0.8, -0.3, -0.5)),
        ("气死我了！", (-0.8, 0.7, 0.5)),
        ("好害怕...", (-0.7, 0.6, -0.6)),
    ]

    for text, (expected_v, expected_a, expected_d) in vad_cases:
        result = analyzer.analyze(text)
        pred_v, pred_a, pred_d = result.vad
        v_diff = abs(pred_v - expected_v)
        print(f"   \"{text[:10]}...\" VAD: ({pred_v:.2f},{pred_a:.2f},{pred_d:.2f}) 误差: V={v_diff:.2f}")


def main():
    """主函数"""
    print_banner()

    parser = argparse.ArgumentParser(
        description="TrueEmotion - 新一代真实情感AI系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python trueemotion_main.py --demo              # 运行演示
  python trueemotion_main.py --analyze "很开心！"  # 分析单条文本
  python trueemotion_main.py --train            # 运行训练
  python trueemotion_main.py --test             # 运行完整测试
        """
    )

    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--analyze", type=str, help="分析单条文本")
    parser.add_argument("--train", action="store_true", help="运行训练")
    parser.add_argument("--test", action="store_true", help="运行完整测试")
    parser.add_argument("--train-samples", type=int, default=2000, help="训练样本数")
    parser.add_argument("--eval-samples", type=int, default=500, help="评估样本数")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")

    args = parser.parse_args()

    # 默认行为：运行演示
    if not any([args.demo, args.analyze, args.train, args.test]):
        args.demo = True

    if args.demo:
        demo_analysis()
        demo_context()
        demo_scenarios()
        print("\n" + "=" * 70)
        print("💡 提示: 使用 --train 运行完整训练，--test 运行完整测试")
        print("=" * 70)

    elif args.analyze:
        analyzer = EmotionAnalyzer()
        analyze_text(analyzer, args.analyze)

    elif args.train:
        run_training(
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            epochs=args.epochs
        )

    elif args.test:
        run_full_test()

    print("\n✨ 完成!")


if __name__ == "__main__":
    main()
