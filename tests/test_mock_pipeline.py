"""End-to-end pipeline tests using mock adapters and runners."""

import numpy as np
from benchlink.schema import CanonicalObs, is_valid_standard_action
from tests.conftest import MockAdapter, MockRunner, MockTactileAdapter


class TestActionLine:
    """Action model pipeline: MockAdapter → MockRunner."""

    def test_mock_adapter_interface(self, mock_adapter):
        """MockAdapter produces valid standard actions."""
        obs = CanonicalObs(
            rgb_static=np.zeros((64, 64, 3), dtype=np.float32),
            proprio=np.zeros(8, dtype=np.float32),
            language="test",
        )
        action = mock_adapter.act(obs)
        assert is_valid_standard_action(action), f"Invalid action: {action}"
        assert action.shape == (7,)

    def test_mock_pipeline(self, mock_adapter, mock_runner):
        """Full pipeline: setup → act → evaluate."""
        result = mock_runner.evaluate(mock_adapter, n_episodes=5)
        assert result["n_episodes"] == 5
        assert len(result["action_mean"]) == 7

    def test_reset_called(self):
        """Adapter.reset() is called each episode in evaluate()."""
        call_count = {"reset": 0}

        class TrackingAdapter(MockAdapter):
            def reset(self):
                call_count["reset"] += 1

        adapter = TrackingAdapter()
        adapter.load("", {"device": "cpu"})

        runner = MockRunner()
        runner.setup({})
        runner.evaluate(adapter, n_episodes=3)
        assert call_count["reset"] == 3


class TestTactileLine:
    """Tactile pipeline: MockTactileAdapter → encode."""

    def test_mock_tactile_encode(self, mock_tactile_adapter):
        """MockTactileAdapter produces feature vectors."""
        img = np.random.randn(224, 224, 3).astype(np.float32)
        features = mock_tactile_adapter.encode(img)
        assert features.shape == (64,)
        assert features.dtype == np.float32

    def test_consistent_encode(self):
        """Same input should give different output (random) — just verify shape."""
        adapter = MockTactileAdapter()
        adapter.load("", {"device": "cpu", "feature_dim": 64})

        img1 = np.zeros((224, 224, 3), dtype=np.float32)
        img2 = np.ones((224, 224, 3), dtype=np.float32)

        f1 = adapter.encode(img1)
        f2 = adapter.encode(img2)
        assert f1.shape == f2.shape == (64,)
