"""
pytest configuration and fixtures
"""
import pytest


@pytest.fixture
def pro(tmp_path):
    """Create TrueEmotionPro instance for testing."""
    from trueemotion import TrueEmotionPro
    return TrueEmotionPro(memory_path=str(tmp_path / "memory"))


@pytest.fixture
def detector():
    """Create HumanEmotionDetector instance for testing."""
    from trueemotion.core.emotions.detector import HumanEmotionDetector
    return HumanEmotionDetector()


@pytest.fixture
def memory_repo(tmp_path):
    """Create MemoryRepository with tmp_path for isolation."""
    from trueemotion.memory.repository import MemoryRepository
    return MemoryRepository(base_path=str(tmp_path / "memory"))


@pytest.fixture
def context():
    """Create ConversationContext instance."""
    from trueemotion.core.analysis.context import ConversationContext
    return ConversationContext()


@pytest.fixture
def contextual_analyzer():
    """Create ContextualAnalyzer instance."""
    from trueemotion.core.analysis.context import ContextualAnalyzer
    return ContextualAnalyzer()


@pytest.fixture
def irony_detector():
    """Create IronyDetector instance."""
    from trueemotion.core.emotions.irony import IronyDetector
    return IronyDetector()


@pytest.fixture
def evolution_manager(memory_repo):
    """Create EvolutionManager with isolated memory repo."""
    from trueemotion.learning.evolution import EvolutionManager
    return EvolutionManager(memory_repo=memory_repo)
