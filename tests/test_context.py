"""
Tests for ConversationContext and ContextualAnalyzer
"""
import pytest

from trueemotion.core.analysis.context import (
    ConversationContext,
    ContextualAnalyzer,
    EmotionTrend,
)


class TestConversationContext:
    """Tests for ConversationContext"""

    def test_add_utterance_and_get_recent_emotions(self, context):
        """Adding utterances populates recent emotions correctly."""
        # Act
        context.add("我很开心", "joy", 0.8)
        context.add("有点难过", "sadness", 0.6)
        context.add("好害怕", "fear", 0.7)

        # Assert
        recent = context.get_recent_emotions(3)
        assert recent == ["joy", "sadness", "fear"]

    def test_get_emotion_trend_rising(self, context):
        """Rising intensity produces a 'rising' trend."""
        # Arrange
        context.add("有点开心", "joy", 0.3)
        context.add("很开心", "joy", 0.6)
        context.add("太开心了", "joy", 0.9)

        # Act
        trend = context.get_emotion_trend()

        # Assert
        assert trend.direction == "rising"
        assert trend.delta > 0.1

    def test_get_emotion_trend_falling(self, context):
        """Falling intensity produces a 'falling' trend."""
        # Arrange
        context.add("非常难过", "sadness", 0.9)
        context.add("有点难过", "sadness", 0.5)
        context.add("好了一些", "sadness", 0.2)

        # Act
        trend = context.get_emotion_trend()

        # Assert
        assert trend.direction == "falling"
        assert trend.delta < -0.1

    def test_get_emotion_trend_stable(self, context):
        """Similar intensity produces a 'stable' trend."""
        # Arrange
        context.add("还好吧", "joy", 0.5)
        context.add("还行吧", "joy", 0.5)

        # Act
        trend = context.get_emotion_trend()

        # Assert
        assert trend.direction == "stable"

    def test_was_emotion_mentioned_recently_true(self, context):
        """Returns True if emotion appeared within N turns."""
        # Arrange
        context.add("我很开心", "joy", 0.8)
        context.add("有点难过", "sadness", 0.5)

        # Assert
        assert context.was_emotion_mentioned_recently("joy", within=3) is True

    def test_was_emotion_mentioned_recently_false(self, context):
        """Returns False if emotion was not seen within window."""
        # Arrange
        context.add("我很开心", "joy", 0.8)

        # Assert
        assert context.was_emotion_mentioned_recently("anger", within=3) is False

    def test_clear_resets_history(self, context):
        """clear() empties the history and resets topic."""
        # Arrange
        context.add("开心", "joy", 0.8)
        context.current_topic = "工作"

        # Act
        context.clear()

        # Assert
        assert len(context.history) == 0
        assert context.current_topic is None

    def test_context_summary_format(self, context):
        """get_context_summary returns expected keys."""
        # Arrange
        context.add("开心", "joy", 0.7)
        context.add("难过", "sadness", 0.4)

        # Act
        summary = context.get_context_summary()

        # Assert
        assert "turns" in summary
        assert "recent_emotions" in summary
        assert "trend" in summary
        assert "trend_delta" in summary
        assert "main_change" in summary
        assert "session_duration_seconds" in summary
        assert summary["turns"] == 2


class TestContextualAnalyzer:
    """Tests for ContextualAnalyzer"""

    def test_analyze_with_context_detects_transition(self, contextual_analyzer):
        """Analyzing two different emotions detects a transition."""
        # Arrange -- first call establishes baseline
        result1 = contextual_analyzer.analyze_with_context("开心", "joy", 0.8)
        assert result1["context_adjustment"] == "new"

        # Act -- second call with different emotion
        result2 = contextual_analyzer.analyze_with_context("难过", "sadness", 0.6)

        # Assert
        assert result2["context_adjustment"] == "turning_worse"
        assert result2["base_emotion"] == "sadness"

    def test_get_follow_up_suggestion_for_reinforced(self, contextual_analyzer):
        """Reinforced emotion with sufficient intensity returns a suggestion."""
        # Arrange -- two sadness rounds to trigger "reinforced"
        contextual_analyzer.analyze_with_context("难过", "sadness", 0.6)
        ctx = contextual_analyzer.analyze_with_context("还是难过", "sadness", 0.5)

        # Act
        suggestion = contextual_analyzer.get_follow_up_suggestion("sadness", 0.5, ctx)

        # Assert
        assert suggestion is not None
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0

    def test_get_follow_up_suggestion_returns_none_for_new(self, contextual_analyzer):
        """A brand-new emotion with low intensity returns None."""
        # Arrange
        ctx = contextual_analyzer.analyze_with_context("还行", "joy", 0.2)

        # Act
        suggestion = contextual_analyzer.get_follow_up_suggestion("joy", 0.2, ctx)

        # Assert
        assert suggestion is None
