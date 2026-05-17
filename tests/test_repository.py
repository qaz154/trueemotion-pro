"""
Tests for MemoryRepository
"""
import pytest

from trueemotion.memory.repository import (
    MemoryRepository,
    UserProfile,
    LearnedPattern,
    STOP_WORDS,
)


class TestGetUser:
    """Tests for get_user behavior"""

    def test_get_user_returns_new_profile_for_unknown_user(self, memory_repo):
        """Unknown user_id returns a fresh UserProfile with defaults."""
        # Act
        profile = memory_repo.get_user("unknown_user")

        # Assert
        assert profile.user_id == "unknown_user"
        assert profile.total_interactions == 0
        assert profile.dominant_emotion is None
        assert profile.relationship_level == 0.0

    def test_save_user_and_get_user_roundtrip(self, memory_repo):
        """Saving then loading a profile preserves all fields."""
        # Arrange
        profile = UserProfile(
            user_id="roundtrip_user",
            total_interactions=5,
            dominant_emotion="joy",
            relationship_level=0.42,
            learned_patterns=3,
            last_emotion="sadness",
            emotional_history=["joy", "sadness"],
            preferred_tone="幽默",
            interaction_style="active",
            emotional_state="波动",
        )

        # Act
        memory_repo.save_user("roundtrip_user", profile)
        loaded = memory_repo.get_user("roundtrip_user")

        # Assert
        assert loaded.user_id == "roundtrip_user"
        assert loaded.total_interactions == 5
        assert loaded.dominant_emotion == "joy"
        assert loaded.relationship_level == 0.42
        assert loaded.learned_patterns == 3
        assert loaded.last_emotion == "sadness"
        assert loaded.emotional_history == ["joy", "sadness"]
        assert loaded.preferred_tone == "幽默"


class TestLearnPattern:
    """Tests for learn_pattern behavior"""

    def test_learn_pattern_creates_pattern_with_keywords(self, memory_repo):
        """learn_pattern returns a pattern that has keywords extracted."""
        # Act
        pattern = memory_repo.learn_pattern(
            user_id="learner",
            emotion="joy",
            response="太棒了，为你感到高兴",
            feedback=0.8,
        )

        # Assert
        assert isinstance(pattern, LearnedPattern)
        assert pattern.emotion == "joy"
        assert pattern.feedback == 0.8
        assert pattern.times_used == 1
        assert len(pattern.keywords) > 0

    def test_learn_pattern_merges_similar_patterns(self, memory_repo):
        """Learning the same emotion+response twice merges into one pattern."""
        # Arrange
        memory_repo.learn_pattern(
            user_id="merger",
            emotion="sadness",
            response="别难过了，会好起来的",
            feedback=0.7,
        )

        # Act
        merged = memory_repo.learn_pattern(
            user_id="merger",
            emotion="sadness",
            response="别难过了，会好起来的",
            feedback=0.8,
        )

        # Assert
        assert merged.times_used == 2
        count = memory_repo.get_pattern_count("merger")
        assert count == 1  # merged, not duplicated


class TestPatternQueries:
    """Tests for pattern counting and listing"""

    def test_get_pattern_count(self, memory_repo):
        """get_pattern_count reflects number of stored patterns."""
        # Arrange
        assert memory_repo.get_pattern_count("counter") == 0
        memory_repo.learn_pattern("counter", "joy", "开心回复", 0.6)
        memory_repo.learn_pattern("counter", "anger", "生气回复", 0.5)

        # Act & Assert
        assert memory_repo.get_pattern_count("counter") == 2

    def test_get_all_patterns(self, memory_repo):
        """get_all_patterns returns patterns keyed by user_id."""
        # Arrange
        memory_repo.learn_pattern("user_a", "joy", "高兴回复", 0.7)
        memory_repo.learn_pattern("user_b", "sadness", "安慰回复", 0.6)

        # Act
        all_patterns = memory_repo.get_all_patterns()

        # Assert
        assert "user_a" in all_patterns
        assert "user_b" in all_patterns
        assert len(all_patterns["user_a"]) == 1
        assert len(all_patterns["user_b"]) == 1


class TestDeleteUser:
    """Tests for delete_user behavior"""

    def test_delete_user_removes_files(self, memory_repo):
        """delete_user removes both user profile and pattern files."""
        # Arrange
        profile = UserProfile(user_id="deleteme", total_interactions=1)
        memory_repo.save_user("deleteme", profile)
        memory_repo.learn_pattern("deleteme", "joy", "回复", 0.5)

        # Verify files exist
        assert memory_repo.get_user("deleteme").total_interactions == 1
        assert memory_repo.get_pattern_count("deleteme") == 1

        # Act
        memory_repo.delete_user("deleteme")

        # Assert -- fresh profile returned, patterns gone
        fresh = memory_repo.get_user("deleteme")
        assert fresh.total_interactions == 0
        assert memory_repo.get_pattern_count("deleteme") == 0


class TestExtractKeywords:
    """Tests for internal _extract_keywords helper"""

    def test_extract_keywords_filters_stop_words(self, memory_repo):
        """Stop words are excluded from extracted keywords."""
        # Act
        keywords = memory_repo._extract_keywords("我很开心因为今天天气好")

        # Assert
        for kw in keywords:
            assert kw not in STOP_WORDS


class TestValidateUserId:
    """Tests for _validate_user_id"""

    def test_validate_user_id_rejects_path_traversal(self, memory_repo):
        """Path traversal characters in user_id raise ValueError."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            memory_repo._validate_user_id("../../../etc/passwd")

    def test_validate_user_id_rejects_special_chars(self, memory_repo):
        """Special characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid user_id"):
            memory_repo._validate_user_id("user;drop table")

    def test_validate_user_id_accepts_valid_ids(self, memory_repo):
        """Valid IDs with alphanumeric, underscores, hyphens, Chinese chars pass."""
        assert memory_repo._validate_user_id("user_123") == "user_123"
        assert memory_repo._validate_user_id("test-user") == "test-user"
        assert memory_repo._validate_user_id("用户甲") == "用户甲"


class TestAtomicWrite:
    """Tests for _atomic_write"""

    def test_atomic_write_survives(self, memory_repo, tmp_path):
        """Atomic write produces a readable file."""
        # Arrange
        target = tmp_path / "memory" / "users" / "atomic_test.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        # Act
        memory_repo._atomic_write(target, '{"key": "value"}')

        # Assert
        assert target.exists()
        import json
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["key"] == "value"


class TestKeywordOverlap:
    """Tests for keyword overlap calculation"""

    def test_keyword_overlap_identical(self, memory_repo):
        """Identical keyword lists yield overlap 1.0."""
        overlap = memory_repo._calculate_keyword_overlap(
            ["开心", "快乐"], ["开心", "快乐"]
        )
        assert overlap == 1.0

    def test_keyword_overlap_disjoint(self, memory_repo):
        """Completely disjoint lists yield overlap 0.0."""
        overlap = memory_repo._calculate_keyword_overlap(
            ["开心", "快乐"], ["难过", "悲伤"]
        )
        assert overlap == 0.0

    def test_keyword_overlap_empty(self, memory_repo):
        """Empty list yields overlap 0.0."""
        assert memory_repo._calculate_keyword_overlap([], ["开心"]) == 0.0
        assert memory_repo._calculate_keyword_overlap(["开心"], []) == 0.0


class TestApplyDecay:
    """Tests for apply_decay behavior"""

    def test_apply_decay_decrements_feedback(self, memory_repo):
        """apply_decay reduces feedback when decay_count reaches threshold."""
        # Arrange -- create a pattern and manually set decay_count=1
        # so the next apply_decay call reaches the >= 2 threshold.
        pattern = memory_repo.learn_pattern("decay_user", "joy", "开心的回复内容", 0.9)

        patterns = memory_repo._load_patterns("decay_user")
        patterns[0].decay_count = 1
        memory_repo._save_patterns("decay_user", patterns)

        # Act -- this call increments to 2, triggering actual decay
        affected = memory_repo.apply_decay()

        # Assert
        assert affected >= 1
        patterns = memory_repo._load_patterns("decay_user")
        assert len(patterns) == 1
        assert patterns[0].feedback < 0.9
