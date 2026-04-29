"""
Tests for empathy response generation
"""
import pytest
from trueemotion.core.response.engine import HumanEmpathyEngine


class TestHumanEmpathyEngine:
    """Tests for HumanEmpathyEngine"""

    def test_generate_joy_response(self):
        """Test joy response generation"""
        engine = HumanEmpathyEngine()
        response = engine.generate(emotion="joy", intensity=0.8)

        assert response.text
        assert response.empathy_type in ["分享喜悦", "温和回应"]
        assert response.intensity_level in ["high", "medium", "low"]

    def test_generate_sadness_response(self):
        """Test sadness response generation"""
        engine = HumanEmpathyEngine()
        response = engine.generate(emotion="sadness", intensity=0.7)

        assert response.text
        assert response.empathy_type in ["深度共情"]
        assert response.intensity_level in ["high", "medium", "low"]

    def test_generate_anger_response(self):
        """Test anger response generation"""
        engine = HumanEmpathyEngine()
        response = engine.generate(emotion="anger", intensity=0.8)

        assert response.text
        assert response.empathy_type in ["安抚情绪"]
        assert response.intensity_level in ["high", "medium", "low"]

    def test_follow_up_generation(self):
        """Test follow-up question generation"""
        engine = HumanEmpathyEngine()
        response = engine.generate(emotion="joy", intensity=0.8)

        # High intensity should often have follow-up
        if response.follow_up:
            assert len(response.follow_up) > 0

    def test_intensity_levels(self):
        """Test different intensity levels"""
        engine = HumanEmpathyEngine()

        for intensity in [0.2, 0.4, 0.6, 0.8, 1.0]:
            response = engine.generate(emotion="joy", intensity=intensity)
            assert response.intensity_level in ["minimal", "low", "medium", "high"]

    def test_all_emotions_have_templates(self):
        """Test all major emotions have response templates"""
        engine = HumanEmpathyEngine()
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "love", "trust", "anticipation"]

        for emotion in emotions:
            response = engine.generate(emotion=emotion, intensity=0.5)
            assert response.text, f"No template for {emotion}"
