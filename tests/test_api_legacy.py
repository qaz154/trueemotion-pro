"""
TrueEmotion Pro v1.15 API 测试
"""
from trueemotion import TrueEmotionPro


def test_basic_analysis():
    """测试基本情感分析"""
    pro = TrueEmotionPro()
    result = pro.analyze("今天太开心了！终于完成了项目！")

    assert result.emotion.primary in ["joy", "ecstasy", "optimism", "anticipation"]
    assert result.emotion.intensity > 0
    assert result.emotion.intensity_label in ["极微", "微弱", "轻微", "中等", "较高", "强烈", "极致"]
    assert result.human_response.text
    print(f"✓ 基本分析: {result.emotion.primary} ({result.emotion.intensity:.3f})")


def test_compound_emotion():
    """测试复合情感检测"""
    pro = TrueEmotionPro()
    result = pro.analyze("又开心又难过，真是悲喜交加啊")

    # 应该检测到joy, sadness 和可能的复合情感
    assert "joy" in result.emotion.all_emotions or "sadness" in result.emotion.all_emotions
    assert result.emotion.emotion_mix  # 应该有情感混合描述
    print(f"✓ 复合情感: {result.emotion.all_emotions}")


def test_continuous_intensity():
    """测试连续强度"""
    pro = TrueEmotionPro()

    # 微弱
    r1 = pro.analyze("有点难过")
    assert r1.emotion.intensity_label in ["极微", "微弱"]

    # 强烈
    r2 = pro.analyze("太开心了！！！终于成功了！！！")
    assert r2.emotion.intensity_label in ["强烈", "极致"]
    print(f"✓ 连续强度: 微弱={r1.emotion.intensity:.3f}, 强烈={r2.emotion.intensity:.3f}")


def test_sadness_detection():
    """测试悲伤情感检测"""
    pro = TrueEmotionPro()
    result = pro.analyze("我太难过了，失恋了，好伤心啊！")

    assert result.emotion.primary in ["sadness", "grief", "despair", "melancholy"]
    assert result.emotion.intensity > 0.1
    print(f"✓ 悲伤检测: {result.emotion.primary}")


def test_anger_detection():
    """测试愤怒情感检测"""
    pro = TrueEmotionPro()
    result = pro.analyze("气死我了！又被骗了！太可恶了！")

    assert result.emotion.primary in ["anger", "rage", "contempt"]
    assert result.emotion.intensity > 0.3
    print(f"✓ 愤怒检测: {result.emotion.primary}")


def test_fear_detection():
    """测试恐惧情感检测"""
    pro = TrueEmotionPro()
    result = pro.analyze("好害怕啊，担心明天的考试...")

    assert result.emotion.primary in ["fear", "terror", "anxiety", "apprehension"]
    print(f"✓ 恐惧检测: {result.emotion.primary}")


def test_despair_detection():
    """测试绝望情感检测"""
    pro = TrueEmotionPro()
    result = pro.analyze("被裁员了，感觉人生没有希望了")

    assert result.emotion.primary in ["despair", "sadness", "grief"]
    print(f"✓ 绝望检测: {result.emotion.primary}")


def test_learning():
    """测试学习功能"""
    pro = TrueEmotionPro()

    result = pro.analyze(
        "工作好累啊，老加班",
        learn=True,
        response="确实累，注意身体啊",
        feedback=0.8,
        user_id="test_user"
    )

    assert result.user_profile.total_interactions >= 1
    print(f"✓ 学习功能: 交互次数={result.user_profile.total_interactions}")


def test_user_profile():
    """测试用户画像"""
    pro = TrueEmotionPro()

    pro.analyze("今天很开心！", user_id="profile_test")
    pro.analyze("工作很累...", user_id="profile_test")

    profile = pro.get_user_profile("profile_test")
    assert profile["user_id"] == "profile_test"
    assert profile["total_interactions"] >= 2
    print(f"✓ 用户画像: {profile['total_interactions']} 次交互")


def test_memory_status():
    """测试记忆状态"""
    pro = TrueEmotionPro()
    pro.analyze("测试文本", user_id="memory_test")

    status = pro.get_memory_status()
    assert "total_users" in status
    assert "total_patterns" in status
    print(f"✓ 记忆状态: {status}")


def test_vad_output():
    """测试VAD输出"""
    pro = TrueEmotionPro()
    result = pro.analyze("太开心了！")

    assert len(result.emotion.vad) == 3
    v, a, d = result.emotion.vad
    assert -1 <= v <= 1
    assert -1 <= a <= 1
    assert -1 <= d <= 1
    print(f"✓ VAD输出: ({v:.2f}, {a:.2f}, {d:.2f})")


def test_empathy_response():
    """测试共情回复"""
    pro = TrueEmotionPro()
    result = pro.analyze("工作好累啊，老加班")

    assert result.human_response.text
    assert result.human_response.empathy_type
    assert result.human_response.intensity_level
    print(f"✓ 共情回复: [{result.human_response.empathy_type}] {result.human_response.text}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("TrueEmotion Pro v1.15 人性化情感系统测试")
    print("=" * 60)

    tests = [
        test_basic_analysis,
        test_compound_emotion,
        test_continuous_intensity,
        test_sadness_detection,
        test_anger_detection,
        test_fear_detection,
        test_despair_detection,
        test_learning,
        test_user_profile,
        test_memory_status,
        test_vad_output,
        test_empathy_response,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
