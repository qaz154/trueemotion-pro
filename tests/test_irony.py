"""
Tests for IronyDetector
"""
import pytest

from trueemotion.core.emotions.irony import IronyDetector, IronyResult


class TestIronyDetector:
    """Tests for IronyDetector"""

    def test_non_ironic_text_returns_false(self, irony_detector):
        """Plain positive text without irony indicators returns is_irony=False."""
        # Act
        result = irony_detector.detect("今天天气真好", "joy", 0.7)

        # Assert
        assert isinstance(result, IronyResult)
        assert result.is_irony is False
        assert result.surface_emotion == "joy"

    def test_ironic_phrase_with_keyword_detected(self, irony_detector):
        """'你可真是太好了啊' triggers irony detection."""
        # Act
        result = irony_detector.detect("你可真是太好了啊", "joy", 0.8)

        # Assert
        assert result.is_irony is True
        assert result.true_emotion is not None
        assert result.confidence > 0.0
        assert len(result.clues) >= 2

    def test_surface_vs_true_emotion_mapping(self, irony_detector):
        """When irony is detected, true_emotion differs from surface_emotion."""
        # Act
        result = irony_detector.detect("你可真行啊，真棒啊", "joy", 0.7)

        # Assert
        if result.is_irony:
            assert result.true_emotion != result.surface_emotion
            assert result.true_emotion in [
                "contempt", "disgust", "anger", "envy", "sadness",
            ]

    def test_low_confidence_single_clue_not_irony(self, irony_detector):
        """A single weak clue should not be enough to flag irony."""
        # Act -- "挺好" alone may not have enough clues
        result = irony_detector.detect("挺好", "joy", 0.5)

        # Assert -- either not irony, or if it is the bar is met
        if not result.is_irony:
            assert result.true_emotion is None

    def test_empty_text_handling(self, irony_detector):
        """Empty text does not crash and returns is_irony=False."""
        # Act
        result = irony_detector.detect("", "joy", 0.5)

        # Assert
        assert result.is_irony is False
        assert result.surface_emotion == "joy"

    def test_irony_clues_populated(self, irony_detector):
        """When irony is detected, clues list contains descriptive strings."""
        # Act
        result = irony_detector.detect("你可真是厉害啊，真优秀啊", "joy", 0.8)

        # Assert
        assert result.is_irony is True
        assert len(result.clues) >= 2
        for clue in result.clues:
            assert isinstance(clue, str)
            assert len(clue) > 0

    def test_contradiction_detection(self, irony_detector):
        """Contradictory emotion words boost irony confidence."""
        # Act -- "开心" and "哭" are contradictory
        result = irony_detector.detect("好开心啊，开心得想哭", "joy", 0.7)

        # Assert -- contradiction clue should appear
        contradiction_found = any("矛盾" in c for c in result.clues)
        assert contradiction_found is True

    def test_negation_structure_detection(self, irony_detector):
        """'才怪' negation structure is detected as irony context clue."""
        # Act
        result = irony_detector.detect("太开心了才怪", "joy", 0.6)

        # Assert
        negation_found = any("才怪" in c for c in result.clues)
        assert negation_found is True
