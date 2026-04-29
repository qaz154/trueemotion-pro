"""
TrueEmotion Pro v4.0 主入口
===========================
新一代中文情感AI系统
"""

import argparse

from trueemotion import TrueEmotionPro


def print_banner():
    """打印横幅"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       TrueEmotion Pro v4.0 - 新一代中文情感AI系统                    ║
║                                                                      ║
║       24种情感 | VAD维度 | 共情回复 | 持续进化 | 长期记忆            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def analyze_text(pro: TrueEmotionPro, text: str, user_id: str = "default"):
    """分析单条文本"""
    result = pro.analyze(text, user_id=user_id)

    print(f"\n📝 输入: {text}")
    print(f"   ├─ 主要情感: {result.emotion.primary}")
    print(f"   ├─ 强度: {result.emotion.intensity:.2f}")
    print(f"   ├─ VAD: ({result.emotion.vad[0]:.2f}, {result.emotion.vad[1]:.2f}, {result.emotion.vad[2]:.2f})")
    print(f"   ├─ 置信度: {result.emotion.confidence:.2f}")
    print(f"   ├─ 共情类型: {result.human_response.empathy_type}")
    print(f"   └─ 回复: {result.human_response.text}")

    if result.human_response.follow_up:
        print(f"       追问: {result.human_response.follow_up}")


def demo_analysis():
    """演示情感分析"""
    print("\n" + "=" * 70)
    print("🎭 情感分析演示")
    print("=" * 70)

    pro = TrueEmotionPro()

    test_texts = [
        ("今天太开心了！终于完成了项目！", "default"),
        ("我很难过，失恋了...", "user1"),
        ("真是气死我了，又被骗了！", "user2"),
        ("好害怕啊，担心明天的考试...", "user3"),
        ("好期待！下个月就要去旅游了！", "user4"),
        ("看着他成功了，既高兴又有点嫉妒。", "user5"),
        ("太为你高兴了！说说怎么回事！", "user6"),
        ("先缓缓，我陪着你", "user7"),
    ]

    for text, user_id in test_texts:
        analyze_text(pro, text, user_id)


def demo_learning():
    """演示学习功能"""
    print("\n" + "=" * 70)
    print("🧠 持续学习演示")
    print("=" * 70)

    pro = TrueEmotionPro()

    # 学习新模式
    print("\n📝 学习新模式...")

    pro.analyze(
        "工作好累啊，老加班",
        learn=True,
        response="确实累，注意身体啊",
        feedback=0.8,
        user_id="learn_demo"
    )

    pro.analyze(
        "今天被老板骂了...",
        learn=True,
        response="心疼你，怎么回事？",
        feedback=0.9,
        user_id="learn_demo"
    )

    pro.analyze(
        "又涨工资了！太开心了！",
        learn=True,
        response="太为你高兴了！说说怎么回事！",
        feedback=0.95,
        user_id="learn_demo"
    )

    # 查看学习结果
    print("\n📊 用户画像:")
    profile = pro.get_user_profile("learn_demo")
    for key, value in profile.items():
        print(f"   ├─ {key}: {value}")

    # 查看记忆状态
    print("\n💾 记忆状态:")
    status = pro.get_memory_status()
    for key, value in status.items():
        print(f"   ├─ {key}: {value}")


def demo_evolve():
    """演示进化功能"""
    print("\n" + "=" * 70)
    print("🔄 进化系统演示")
    print("=" * 70)

    pro = TrueEmotionPro()

    # 先学习一些模式
    print("\n📝 学习新模式...")
    for i in range(5):
        pro.analyze(
            "工作好累啊",
            learn=True,
            response="确实累，注意身体啊",
            feedback=0.8,
            user_id="evolve_demo"
        )

    # 执行进化
    print("\n🔄 执行进化...")
    result = pro.evolve()
    print(f"   ├─ 分析模式数: {result['total_patterns_analyzed']}")
    print(f"   ├─ 涉及情感数: {result['emotions_with_patterns']}")
    print(f"   └─ 进化规则数: {len(result['evolved_rules'])}")

    # 查看进化状态
    print("\n📊 进化状态:")
    status = pro.get_evolution_status()
    for key, value in status.items():
        print(f"   ├─ {key}: {value}")


def run_tests():
    """运行内置测试"""
    print("\n" + "=" * 70)
    print("🧪 情感识别测试")
    print("=" * 70)

    pro = TrueEmotionPro()

    test_cases = [
        ("太开心了！", "joy"),
        ("很难过...", "sadness"),
        ("气死我了！", "anger"),
        ("好害怕...", "fear"),
        ("好期待！", "anticipation"),
        ("惊呆了！", "surprise"),
        ("真恶心！", "disgust"),
        ("我爱你！", "love"),
    ]

    correct = 0
    for text, expected in test_cases:
        result = pro.analyze(text)
        pred = result.emotion.primary
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"   {status} \"{text[:15]}\" → 期望:{expected}, 预测:{pred}")

    print(f"\n   情感分类准确率: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")


def main():
    """主函数"""
    print_banner()

    parser = argparse.ArgumentParser(
        description="TrueEmotion Pro v4.0 - 新一代中文情感AI系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--demo-learning", action="store_true", help="演示学习功能")
    parser.add_argument("--demo-evolve", action="store_true", help="演示进化功能")
    parser.add_argument("--analyze", type=str, help="分析单条文本")
    parser.add_argument("--user-id", type=str, default="default", help="用户ID")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--stats", action="store_true", help="显示系统统计")

    args = parser.parse_args()

    # 默认行为：运行演示
    if not any([args.demo, args.demo_learning, args.demo_evolve, args.analyze, args.test, args.stats]):
        args.demo = True

    if args.demo:
        demo_analysis()

    elif args.demo_learning:
        demo_learning()

    elif args.demo_evolve:
        demo_evolve()

    elif args.analyze:
        pro = TrueEmotionPro()
        analyze_text(pro, args.analyze, args.user_id)

    elif args.test:
        run_tests()

    elif args.stats:
        pro = TrueEmotionPro()
        print("\n📊 系统统计:")
        stats = pro.get_stats()
        for key, value in stats.items():
            print(f"   ├─ {key}: {value}")

    print("\n✨ 完成!")


if __name__ == "__main__":
    main()
