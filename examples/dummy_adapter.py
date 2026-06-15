"""
DummyAdapter — a zero-dependency ModelAdapter example.

No external dependencies beyond numpy. Run it directly:
    python examples/01_dummy_adapter.py
"""

import numpy as np
from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs, is_valid_standard_action


class DummyAdapter(ModelAdapter):
    """A minimal ModelAdapter that outputs a fixed action.

    Demonstrates the three required methods: load(), act(), reset().
    """

    def load(self, checkpoint: str, config: dict) -> None:
        """Pretend to load weights — just stores config params."""
        self.device = config.get("device", "cpu")
        # Return a fixed dummy action: [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5]
        self._action = np.array(config.get("dummy_action",
            [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5]), dtype=np.float64)
        print(f"[DummyAdapter] Loaded (device={self.device})")

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Ignore the observation and return the fixed dummy action."""
        return self._action.copy()

    def reset(self) -> None:
        """Reset internal state (no-op for a dummy adapter)."""
        print("[DummyAdapter] Reset")


# ── Standalone smoke test ──
if __name__ == "__main__":
    adapter = DummyAdapter()
    adapter.load("", {"device": "cpu"})

    # Create a minimal observation
    obs = CanonicalObs(
        rgb_static=np.random.rand(224, 224, 3).astype(np.float32),
        proprio=np.zeros(8, dtype=np.float32),
        language="pick up the object",
    )

    action = adapter.act(obs)
    assert is_valid_standard_action(action), f"Invalid action: {action}"
    print(f"[DummyAdapter] action={action}")
    print("[DummyAdapter] Valid standard action: True")

    adapter.reset()
    print("[DummyAdapter] All checks passed!")