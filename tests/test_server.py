"""
Tests for FastAPI server endpoints
"""
import pytest
from fastapi.testclient import TestClient

from trueemotion.api.server import app
from trueemotion import TrueEmotionPro


@pytest.fixture
def client(tmp_path):
    """Create a TestClient with an isolated TrueEmotionPro instance."""
    app.state.pro = TrueEmotionPro(memory_path=str(tmp_path / "memory"))
    with TestClient(app) as c:
        yield c
    app.state.pro = None


class TestInfoEndpoints:
    """Tests for informational endpoints"""

    def test_root_returns_service_info(self, client):
        """GET / returns service name and version."""
        # Act
        resp = client.get("/")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TrueEmotion Pro"
        assert "version" in data
        assert data["docs"] == "/docs"

    def test_health_returns_healthy(self, client):
        """GET /health returns healthy status."""
        # Act
        resp = client.get("/health")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "engine" in data

    def test_stats_returns_stats(self, client):
        """GET /stats returns system statistics."""
        # Act
        resp = client.get("/stats")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        stats = data["data"]
        assert "total_users" in stats
        assert "total_patterns" in stats


class TestAnalyzeEndpoints:
    """Tests for analysis endpoints"""

    def test_analyze_with_valid_text(self, client):
        """POST /analyze with valid text returns emotion data."""
        # Act
        resp = client.post("/analyze", json={"text": "今天好开心啊！"})

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        emotion = data["data"]["emotion"]
        assert "primary" in emotion
        assert "intensity" in emotion
        assert "vad" in emotion
        assert "confidence" in emotion

    def test_analyze_with_empty_text_returns_422(self, client):
        """POST /analyze with empty text returns 422 validation error."""
        # Act
        resp = client.post("/analyze", json={"text": ""})

        # Assert
        assert resp.status_code == 422

    def test_analyze_batch(self, client):
        """POST /analyze/batch with multiple texts returns list of results."""
        # Act
        resp = client.post(
            "/analyze/batch",
            json={"texts": ["开心", "难过"], "user_id": "batch_user"},
        )

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["data"]) == 2


class TestUserEndpoints:
    """Tests for user-related endpoints"""

    def test_get_user_profile(self, client):
        """GET /profile/{user_id} returns user profile data."""
        # Arrange -- create interaction so user exists
        client.post("/analyze", json={"text": "测试", "user_id": "profile_user"})

        # Act
        resp = client.get("/profile/profile_user")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["user_id"] == "profile_user"


class TestMemoryEndpoints:
    """Tests for memory endpoints"""

    def test_memory_status(self, client):
        """GET /memory/status returns memory statistics."""
        # Act
        resp = client.get("/memory/status")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "total_users" in data["data"]

    def test_memory_reset(self, client):
        """POST /memory/reset clears user data."""
        # Arrange
        client.post("/analyze", json={"text": "测试", "user_id": "reset_user"})

        # Act
        resp = client.post("/memory/reset", json={"user_id": "reset_user"})

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


class TestEvolutionEndpoints:
    """Tests for evolution endpoints"""

    def test_evolve(self, client):
        """POST /evolve triggers evolution and returns result."""
        # Act
        resp = client.post("/evolve")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "total_patterns_analyzed" in data["data"]
        assert "evolved_rules" in data["data"]
