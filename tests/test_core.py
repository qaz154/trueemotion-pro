"""
pytest configuration and fixtures
"""
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def pro():
    """Create TrueEmotionPro instance for testing."""
    from trueemotion import TrueEmotionPro
    return TrueEmotionPro(memory_path="./test_memory")


@pytest.fixture
def detector():
    """Create HumanEmotionDetector instance for testing."""
    from trueemotion.core.emotions.detector import HumanEmotionDetector
    return HumanEmotionDetector()


@pytest.fixture(autouse=True)
def cleanup_test_memory():
    """Clean up test memory after each test."""
    yield
    import shutil
    test_mem = Path("./test_memory")
    if test_mem.exists():
        shutil.rmtree(test_mem)
