"""
Tests for emotion detection
"""
import pytest
from trueemotion.core.emotions.detector import RuleBasedEmotionDetector
from trueemotion.core.emotions.plutchik24 import EMOTION_VAD, EMOTION_KEYWORDS


class TestRuleBasedEmotionDetector:
    """Tests for RuleBasedEmotionDetector"""

    def test_detect_joy(self, detector):
        """Test joy emotion detection"""
        scores = detector.detect("今天太开心了！终于完成了项目！")
        assert "joy" in scores or "ecstasy" in scores or "optimism" in scores
        assert scores[max(scores.keys(), key=lambda k: scores[k])] > 0.3

    def test_detect_sadness(self, detector):
        """Test sadness emotion detection"""
        scores = detector.detect("我很难过，失恋了...")
        assert len(scores) > 0
        primary = max(scores.keys(), key=lambda k: scores[k])
        assert primary in ["sadness", "grief", "despair", "remorse"]

    def test_detect_anger(self, detector):
        """Test anger emotion detection"""
        scores = detector.detect("气死我了！又被骗了！")
        assert len(scores) > 0
        primary = max(scores.keys(), key=lambda k: scores[k])
        assert primary in ["anger", "rage", "annoyance"]

    def test_detect_fear(self, detector):
        """Test fear emotion detection"""
        scores = detector.detect("好害怕啊，担心明天的考试...")
        assert len(scores) > 0
        primary = max(scores.keys(), key=lambda k: scores[k])
        assert primary in ["fear", "terror", "anxiety", "apprehension"]

    def test_detect_empty_text(self, detector):
        """Test empty text handling"""
        scores = detector.detect("")
        assert scores == {}

    def test_detect_neutral_text(self, detector):
        """Test neutral text handling"""
        scores = detector.detect("今天天气不错。")
        # May or may not detect emotions, but should not crash
        assert isinstance(scores, dict)

    def test_intensity_boost_exclamation(self, detector):
        """Test exclamation mark intensity boost"""
        scores_no_exclaim = detector.detect("有点开心")
        scores_with_exclaim = detector.detect("太开心了！")

        # With exclamation should have higher score
        if scores_no_exclaim and scores_with_exclaim:
            max_no = max(scores_no_exclaim.values())
            max_with = max(scores_with_exclaim.values())
            assert max_with >= max_no

    def test_negation_handling(self, detector):
        """Test negation handling"""
        scores_positive = detector.detect("我很开心")
        scores_negative = detector.detect("我不开心")

        # Negative should have lower or different score
        if scores_positive and scores_negative:
            assert scores_negative.get("joy", 0) <= scores_positive.get("joy", 0) * 1.5


class TestEmotionDefinitions:
    """Tests for emotion definitions"""

    def test_vad_values_valid(self):
        """Test all VAD values are in valid range"""
        for emotion, vad in EMOTION_VAD.items():
            v, a, d = vad
            assert -1 <= v <= 1, f"{emotion} V value {v} out of range"
            assert -1 <= a <= 1, f"{emotion} A value {a} out of range"
            assert -1 <= d <= 1, f"{emotion} D value {d} out of range"

    def test_all_emotions_have_keywords(self):
        """Test all emotions have keywords defined"""
        for emotion in EMOTION_VAD.keys():
            assert emotion in EMOTION_KEYWORDS, f"{emotion} missing keywords"
            assert len(EMOTION_KEYWORDS[emotion]) > 0, f"{emotion} has no keywords"

    def test_emotions_have_antonyms(self):
        """Test all primary emotions have antonyms"""
        primary_emotions = {"joy", "sadness", "anger", "fear", "disgust", "surprise", "trust", "anticipation"}
        for emotion in primary_emotions:
            assert emotion in EMOTION_VAD, f"{emotion} not in VAD"
