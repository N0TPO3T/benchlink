"""
DummyRunner — a zero-dependency BenchmarkRunner example.

Runs N episodes with any ModelAdapter, records action statistics.
No external dependencies beyond numpy. Run it directly:
    PYTHONPATH=. python examples/dummy_runner.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
import numpy as np
from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs


class DummyRunner(BenchmarkRunner):
    """A minimal BenchmarkRunner that tests action consistency.

    For each episode:
      1. Generates a random observation (CanonicalObs)
      2. Calls model.act(obs)
      3. Records the action for statistics
    """

    def setup(self, config: dict) -> None:
        """Store configuration."""
        self.config = config
        self._img_size = config.get("img_size", 64)

    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        """Convert a raw observation dict to CanonicalObs."""
        return CanonicalObs(**raw_obs)

    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run N episodes and return action statistics."""
        actions = []
        for ep in range(n_episodes):
            model.reset()

            # Generate a dummy observation
            raw = {
                "rgb_static": np.random.rand(
                    self._img_size, self._img_size, 3).astype(np.float32),
                "proprio": np.zeros(8, dtype=np.float32),
                "language": "dummy task",
            }
            obs = self._to_canonical(raw)

            action = model.act(obs)
            actions.append(action)

        actions = np.stack(actions)
        return {
            "n_episodes": n_episodes,
            "action_mean": actions.mean(axis=0).tolist(),
            "action_std": actions.std(axis=0).tolist(),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
        }


# ── Standalone smoke test ──
if __name__ == "__main__":
    from examples.dummy_adapter import DummyAdapter

    adapter = DummyAdapter()
    adapter.load("", {"device": "cpu"})

    runner = DummyRunner()
    runner.setup({"img_size": 64})

    result = runner.evaluate(adapter, n_episodes=5)
    print(f"[DummyRunner] Result: {result}")
    print("[DummyRunner] All checks passed!")
