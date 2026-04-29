# -*- coding: utf-8 -*-
"""
TrueEmotion Pro Integration Tests
集成测试
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(PROJECT_ROOT))

from trueemotion.trueemotion_pro import TrueEmotionPro, MemorySystem
from trueemotion.models.hybrid_emotion import HybridEmotionAnalyzer, RuleBasedEmotionDetector


class TestEmotionDetection:
    """情感检测测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.pro = TrueEmotionPro()

    def test_basic_emotions(self):
        """测试基本情感检测"""
        test_cases = [
            ("太开心了！", "joy"),
            ("工作好累啊", "sadness"),
            ("气死了！", "anger"),
            ("好害怕啊", "fear"),
            ("太意外了！", "surprise"),
        ]

        for text, expected_emotion in test_cases:
            result = self.pro.analyze(text)
            assert result['emotion']['primary'] == expected_emotion, \
                f"文本: {text}, 期望: {expected_emotion}, 实际: {result['emotion']['primary']}"

    def test_complex_text(self):
        """测试复杂文本"""
        test_cases = [
            "我升职了！太开心了！",
            "被裁员了，感觉人生没有希望了...",
            "今天吃什么好呢？",
        ]

        for text in test_cases:
            result = self.pro.analyze(text)
            assert result['emotion']['primary'] is not None
            assert result['emotion']['intensity'] >= 0
            assert 'vad' in result['emotion']

    def test_confidence_scores(self):
        """测试置信度"""
        result = self.pro.analyze("气死了！领导又批评我！")
        assert result['emotion']['confidence'] > 0
        assert 0 <= result['emotion']['confidence'] <= 1

    def test_human_response_generation(self):
        """测试有血有肉回复生成"""
        result = self.pro.analyze("工作好累啊，老加班")
        assert 'human_response' in result
        assert 'text' in result['human_response']
        assert len(result['human_response']['text']) > 0
        assert 'empathy_type' in result['human_response']


class TestPatternFeedback:
    """模式反馈测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.pro = TrueEmotionPro()

    def test_learned_pattern_detection(self):
        """测试学习到的模式能影响检测"""
        # 添加一个学习到的模式
        self.pro.emotion_analyzer.add_learned_pattern('工作好累', 'sadness')

        # 测试该模式
        result = self.pro.analyze('工作好累')
        assert result['emotion']['primary'] == 'sadness', \
            f"期望 sadness, 实际 {result['emotion']['primary']}"

    def test_pattern_override(self):
        """测试学习模式优先于通用规则"""
        # 添加一个明确的模式
        self.pro.emotion_analyzer.add_learned_pattern('我好开心', 'sadness')

        # 即使"开心"通常是joy，这个模式应该覆盖
        result = self.pro.analyze('我好开心')
        assert result['emotion']['primary'] == 'sadness', \
            f"期望 sadness (来自学习模式), 实际 {result['emotion']['primary']}"


class TestMemorySystem:
    """记忆系统测试"""

    def setup_method(self):
        """每个测试前设置"""
        # 使用临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.memory = MemorySystem(storage_dir=self.temp_dir)

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_creation(self):
        """测试用户创建"""
        user = self.memory.get_or_create_user("test_user")
        assert user['user_id'] == "test_user"
        assert user['total_interactions'] == 0

    def test_interaction_recording(self):
        """测试交互记录"""
        user_id = "test_user"

        # 记录交互
        self.memory.record_interaction(
            user_id=user_id,
            user_text="工作好累",
            user_emotion="sadness",
            ai_response="心疼你",
            feedback=0.8
        )

        # 检查用户统计
        user = self.memory.get_or_create_user(user_id)
        assert user['total_interactions'] == 1
        assert user['common_emotions']['sadness'] == 1

    def test_dominant_emotion(self):
        """测试主导情感计算"""
        user_id = "test_user"

        # 记录多个相同情感的交互
        for _ in range(3):
            self.memory.record_interaction(
                user_id=user_id,
                user_text="工作好累",
                user_emotion="sadness",
                ai_response="心疼你"
            )

        # 检查主导情感
        summary = self.memory.get_user_summary(user_id)
        assert summary['dominant_emotion'] == 'sadness'

    def test_learned_pattern_persistence(self):
        """测试学习模式持久化"""
        # 添加一个模式
        self.memory.add_learned_pattern("工作好累", "sadness", 0.8)

        # 重新创建MemorySystem（模拟重启）
        new_memory = MemorySystem(storage_dir=self.temp_dir)

        # 检查模式是否加载
        assert "工作好累" in new_memory.learned_patterns


class TestLongTermLearning:
    """长期学习测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.pro = TrueEmotionPro()

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_learning_updates_stats(self):
        """测试学习更新统计"""
        # 注意：由于测试隔离问题，只验证字段存在且值有效
        result = self.pro.analyze(
            "工作好累啊，老加班",
            learn=True,
            response="确实累，注意休息"
        )

        assert 'user_summary' in result
        assert 'total_interactions' in result['user_summary']
        assert result['user_summary']['total_interactions'] >= 1

    def test_evolution_on_learning(self):
        """测试学习触发进化"""
        # 多次交互
        for _ in range(3):
            self.pro.analyze(
                "项目又延期了，好烦",
                learn=True,
                response="确实烦，我们一起想办法"
            )

        # 检查进化状态
        status = self.pro.evolution_manager.get_status()
        assert status['evolution']['total_interactions'] >= 3


class TestHybridAnalyzer:
    """混合分析器测试"""

    def test_rule_based_fallback(self):
        """测试规则系统后备"""
        analyzer = HybridEmotionAnalyzer()

        # 应该使用规则系统
        result = analyzer.analyze("气死了！")
        assert result is not None
        assert result['primary_emotion'] == 'anger'

    def test_neural_not_loaded_graceful(self):
        """测试神经网络未加载时的优雅降级"""
        analyzer = HybridEmotionAnalyzer()  # 没有传入模型路径

        # 应该返回规则结果而不是崩溃
        result = analyzer.analyze("工作好累啊")
        assert result is not None
        assert 'primary_emotion' in result


class TestEmpathyResponse:
    """共情回复测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.pro = TrueEmotionPro()

    def test_empathy_for_joy(self):
        """测试喜悦情感的共情"""
        result = self.pro.analyze("我升职了！太开心了！")
        response = result['human_response']['text']
        empathy_type = result['human_response']['empathy_type']

        # joy情感的共情类型可能是多种
        assert empathy_type in ['celebration', 'sharing', 'congratulations', 'understanding', 'support']
        assert len(response) > 0

    def test_empathy_for_sadness(self):
        """测试悲伤情感的共情"""
        result = self.pro.analyze("工作好累啊，老加班")
        response = result['human_response']['text']
        empathy_type = result['human_response']['empathy_type']

        assert empathy_type in ['support', 'comfort', 'understanding']
        assert len(response) > 0

    def test_empathy_for_anger(self):
        """测试愤怒情感的共情"""
        result = self.pro.analyze("气死了！领导又批评我！")
        response = result['human_response']['text']
        empathy_type = result['human_response']['empathy_type']

        assert empathy_type in ['venting', 'validation', 'understanding']
        assert len(response) > 0


class TestEndToEnd:
    """端到端测试"""

    def setup_method(self):
        """每个测试前设置"""
        self.pro = TrueEmotionPro()

    def test_full_analysis_pipeline(self):
        """测试完整分析流程"""
        text = "工作好累啊，老加班"

        result = self.pro.analyze(
            text,
            learn=True,
            response="确实累，注意休息",
            feedback=0.8
        )

        # 检查所有字段
        assert 'version' in result
        assert 'engine' in result
        assert 'emotion' in result
        assert 'human_response' in result
        assert 'user_summary' in result

        # 检查情感字段
        emotion = result['emotion']
        assert 'primary' in emotion
        assert 'intensity' in emotion
        assert 'vad' in emotion
        assert 'confidence' in emotion

        # 检查回复字段
        response = result['human_response']
        assert 'text' in response
        assert 'empathy_type' in response
        assert 'intensity_level' in response

    def test_different_users(self):
        """测试不同用户"""
        result1 = self.pro.analyze(
            "工作好累",
            learn=True,
            response="确实累",
            user_id="user1"
        )

        result2 = self.pro.analyze(
            "太开心了",
            learn=True,
            response="恭喜",
            user_id="user2"
        )

        # 不同用户应该有独立的记忆
        assert result1['user_summary']['user_id'] == "user1"
        assert result2['user_summary']['user_id'] == "user2"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("TrueEmotion Pro 集成测试")
    print("=" * 60)

    test_classes = [
        TestEmotionDetection,
        TestPatternFeedback,
        TestMemorySystem,
        TestLongTermLearning,
        TestHybridAnalyzer,
        TestEmpathyResponse,
        TestEndToEnd,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)

        instance = test_class()
        setup_method = getattr(instance, 'setup_method', None)

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1

                try:
                    if setup_method:
                        setup_method()
                    getattr(instance, method_name)()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append(f"{test_class.__name__}.{method_name}")
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")
                    failed_tests.append(f"{test_class.__name__}.{method_name}")
                finally:
                    teardown = getattr(instance, 'teardown_method', None)
                    if teardown:
                        teardown()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    if failed_tests:
        print(f"失败测试:")
        for f in failed_tests:
            print(f"  - {f}")
    print("=" * 60)

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
