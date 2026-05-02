"""
Tests for EmotionAnalyzer
"""
import pytest
from trueemotion.core.analysis.analyzer import EmotionAnalyzer, AnalyzeOptions


class TestEmotionAnalyzer:
    """Tests for EmotionAnalyzer"""

    def test_analyze_basic(self, pro):
        """Test basic analysis"""
        result = pro.analyze("今天太开心了！")

        assert result.version == "1.15"
        assert result.engine in ["rule-v1.15", "llm-v1.15"]
        assert result.emotion.primary in ["joy", "ecstasy", "optimism", "anticipation"]
        assert result.emotion.intensity > 0
        assert len(result.emotion.vad) == 3
        assert result.human_response.text

    def test_analyze_with_options(self, pro):
        """Test analysis with options"""
        result = pro.analyze(
            "工作好累啊",
            user_id="test_user",
            learn=True,
            response="确实累",
            feedback=0.8,
        )

        assert result.user_profile.user_id == "test_user"
        assert result.user_profile.total_interactions >= 1

    def test_analyze_returns_valid_vad(self, pro):
        """Test VAD values are valid"""
        result = pro.analyze("太开心了！")

        v, a, d = result.emotion.vad
        assert -1 <= v <= 1
        assert -1 <= a <= 1
        assert -1 <= d <= 1

    def test_analyze_returns_confidence(self, pro):
        """Test confidence is calculated"""
        result = pro.analyze("太开心了！")

        assert 0 <= result.emotion.confidence <= 1

    def test_get_user_profile(self, pro):
        """Test getting user profile"""
        # Create some interactions
        pro.analyze("很开心！", user_id="profile_user")
        pro.analyze("很难过...", user_id="profile_user")

        profile = pro.get_user_profile("profile_user")

        assert profile["user_id"] == "profile_user"
        assert profile["total_interactions"] >= 2

    def test_get_stats(self, pro):
        """Test getting stats"""
        pro.analyze("测试", user_id="stats_user")

        stats = pro.get_stats()

        assert "total_users" in stats
        assert "total_patterns" in stats
        assert stats["total_users"] >= 1

    def test_evolve(self, pro):
        """Test evolution"""
        result = pro.evolve()

        assert "total_patterns_analyzed" in result
        assert "emotions_with_patterns" in result
        assert "evolved_rules" in result


class TestAnalyzeOptions:
    """Tests for AnalyzeOptions"""

    def test_default_options(self):
        """Test default options"""
        options = AnalyzeOptions()

        assert options.learn is False
        assert options.user_id == "default"
        assert options.feedback == 0.5

    def test_custom_options(self):
        """Test custom options"""
        options = AnalyzeOptions(
            user_id="custom",
            learn=True,
            response="回复",
            feedback=0.9,
            context="上下文",
        )

        assert options.user_id == "custom"
        assert options.learn is True
        assert options.response == "回复"
        assert options.feedback == 0.9
        assert options.context == "上下文"
