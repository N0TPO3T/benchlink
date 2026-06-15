"""Shared pytest fixtures: MockAdapter, MockRunner, MockTactileAdapter."""

from typing import Dict, Any
import numpy as np
import pytest

from benchlink.base import ModelAdapter, TactileAdapter, BenchmarkRunner
from benchlink.schema import CanonicalObs


class MockAdapter(ModelAdapter):
    """ModelAdapter that returns a fixed action for testing."""

    def load(self, checkpoint: str, config: dict) -> None:
        self.device = config.get("device", "cpu")
        self._action = np.array(config.get(
            "dummy_action", [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5]
        ), dtype=np.float64)

    def act(self, obs: CanonicalObs) -> np.ndarray:
        return self._action.copy()

    def reset(self) -> None:
        pass


class MockTactileAdapter(TactileAdapter):
    """TactileAdapter that returns random features for testing."""

    def load(self, checkpoint: str, config: dict) -> None:
        self.device = config.get("device", "cpu")
        self.dim = config.get("feature_dim", 64)

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        return np.random.randn(self.dim).astype(np.float32)


class MockRunner(BenchmarkRunner):
    """BenchmarkRunner that records actions for testing."""

    def setup(self, config: dict) -> None:
        self.config = config

    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        return CanonicalObs(**raw_obs)

    def evaluate(
        self, model: ModelAdapter, n_episodes: int = 10, **kwargs
    ) -> Dict[str, Any]:
        actions = []
        for _ in range(n_episodes):
            model.reset()
            obs = CanonicalObs(
                rgb_static=np.zeros((64, 64, 3), dtype=np.float32),
                proprio=np.zeros(8, dtype=np.float32),
            )
            action = model.act(obs)
            actions.append(action)

        actions = np.stack(actions)
        return {
            "n_episodes": n_episodes,
            "action_mean": actions.mean(axis=0).tolist(),
        }


@pytest.fixture
def mock_adapter():
    adapter = MockAdapter()
    adapter.load("", {"device": "cpu"})
    return adapter


@pytest.fixture
def mock_tactile_adapter():
    adapter = MockTactileAdapter()
    adapter.load("", {"device": "cpu", "feature_dim": 64})
    return adapter


@pytest.fixture
def mock_runner():
    runner = MockRunner()
    runner.setup({"img_size": 64})
    return runner
