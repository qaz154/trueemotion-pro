"""
Tests for EvolutionManager
"""
import pytest

from trueemotion.learning.evolution import EvolutionManager
from trueemotion.memory.repository import MemoryRepository, LearnedPattern


class TestEvolveEmpty:
    """Tests for evolve with no data"""

    def test_evolve_with_no_patterns_returns_empty(self, evolution_manager):
        """evolve() on an empty repo returns zero patterns and rules."""
        # Act
        result = evolution_manager.evolve()

        # Assert
        assert result["total_patterns_analyzed"] == 0
        assert result["emotions_with_patterns"] == 0
        assert result["evolved_rules"] == []


class TestEvolveWithPatterns:
    """Tests for evolve with data"""

    def test_evolve_with_high_feedback_patterns_creates_rules(
        self, memory_repo, evolution_manager
    ):
        """High-feedback patterns meeting threshold produce evolved rules."""
        # Arrange -- create enough high-feedback patterns for one emotion
        for i in range(3):
            memory_repo.learn_pattern(
                user_id=f"evo_user_{i}",
                emotion="joy",
                response=f"太棒了回复{i}",
                feedback=0.85,
            )

        # Act
        result = evolution_manager.evolve()

        # Assert
        assert result["total_patterns_analyzed"] == 3
        assert result["emotions_with_patterns"] >= 1
        assert len(result["evolved_rules"]) >= 1
        joy_rule = next(
            (r for r in result["evolved_rules"] if r["emotion"] == "joy"), None
        )
        assert joy_rule is not None
        assert joy_rule["source_patterns"] >= 2
        assert joy_rule["avg_feedback"] >= 0.6


class TestConfidenceCalculation:
    """Tests for confidence formula"""

    def test_confidence_calculation_formula(self, evolution_manager):
        """Confidence uses weighted average of pattern count, feedback, usage."""
        # Act
        confidence = evolution_manager._calculate_confidence(
            pattern_count=5,
            avg_feedback=0.8,
            total_usage=20,
        )

        # Assert
        # pattern_conf = min(1.0, 5/5) = 1.0
        # feedback_conf = 0.8
        # usage_conf = min(1.0, 20/20) = 1.0
        # expected = 1.0 * 0.3 + 0.8 * 0.5 + 1.0 * 0.2 = 0.9
        assert abs(confidence - 0.9) < 0.01

    def test_confidence_low_data(self, evolution_manager):
        """Low pattern count and usage produce lower confidence."""
        # Act
        confidence = evolution_manager._calculate_confidence(
            pattern_count=1,
            avg_feedback=0.5,
            total_usage=1,
        )

        # Assert
        assert confidence < 0.5


class TestEvolutionHistory:
    """Tests for evolution history tracking"""

    def test_evolution_history_tracked(self, evolution_manager):
        """Each evolve() call adds a history entry."""
        # Arrange
        assert len(evolution_manager._evolution_history) == 0

        # Act
        evolution_manager.evolve()
        evolution_manager.evolve()

        # Assert
        assert len(evolution_manager._evolution_history) == 2

    def test_get_evolution_status_format(self, memory_repo, evolution_manager):
        """get_evolution_status returns expected keys and types."""
        # Act
        status = evolution_manager.get_evolution_status()

        # Assert
        assert "total_patterns" in status
        assert "high_quality_patterns" in status
        assert "very_high_quality_patterns" in status
        assert "total_usage_count" in status
        assert "min_feedback_threshold" in status
        assert "ready_to_evolve" in status
        assert "evolution_count" in status
        assert isinstance(status["ready_to_evolve"], bool)


class TestKeywordExtraction:
    """Tests for keyword extraction from patterns"""

    def test_extract_keywords_from_patterns(self, memory_repo, evolution_manager):
        """_extract_keywords_advanced scores keywords by usage * feedback."""
        # Arrange
        patterns = [
            LearnedPattern(
                user_id="u1",
                emotion="joy",
                response="回复1",
                feedback=0.9,
                times_used=3,
                keywords=["开心", "快乐"],
            ),
            LearnedPattern(
                user_id="u2",
                emotion="joy",
                response="回复2",
                feedback=0.8,
                times_used=2,
                keywords=["开心", "幸福"],
            ),
        ]

        # Act
        keywords = evolution_manager._extract_keywords_advanced(patterns)

        # Assert
        assert "开心" in keywords  # appears in both patterns, highest score
        assert len(keywords) > 0
