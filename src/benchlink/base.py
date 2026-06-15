"""
Core abstract base classes: ModelAdapter, TactileAdapter, BenchmarkRunner.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from .schema import CanonicalObs


class ModelAdapter(ABC):
    """Action model adapter — encapsulates baseline inference.

    Subclasses must implement:
        load()        — load weights
        act(obs)      — single-step inference, return standard action (7,)
        reset()       — reset internal state (episode boundary)
    """

    def __init__(self):
        self.device: str = "cpu"
        self.model = None

    @abstractmethod
    def load(self, checkpoint: str, config: dict) -> None:
        """Load model weights and configuration."""

    @abstractmethod
    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Receive standard observation, return standard action ndarray shape=(7,)."""

    @abstractmethod
    def reset(self) -> None:
        """Called at each episode boundary; clears rollout buffer / history."""


class TactileAdapter(ABC):
    """Tactile encoder adapter — encapsulates tactile model encode pipeline.

    Subclasses must implement:
        load()               — load weights
        encode(tactile_img)  — tactile image -> feature vector
    """

    def __init__(self):
        self.device: str = "cpu"
        self.model = None

    @abstractmethod
    def load(self, checkpoint: str, config: dict) -> None:
        """Load model weights and configuration."""

    @abstractmethod
    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector."""


class BenchmarkRunner(ABC):
    """Benchmark evaluation executor — wraps simulation environment or offline dataset.

    Subclasses must implement:
        setup(config)            — initialize environment and configuration
        evaluate(model, n_eps)   — run evaluation and return metrics
        _to_canonical(raw_obs)   — native observation -> standard observation dict
        _from_canonical(action)  — standard action -> native action
    """

    def __init__(self):
        self.config: dict = {}

    @abstractmethod
    def setup(self, config: dict) -> None:
        """Initialize benchmark (load dataset / create environment)."""

    @abstractmethod
    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run N episodes, return evaluation metrics dict.

        Typical metrics:
            success_rate       — success rate (simulation)
            position_error     — position error (offline replay)
            rotation_error     — rotation error (offline replay)
        """

    @abstractmethod
    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        """Benchmark native observation -> CanonicalObs."""

    def _from_canonical(self, action: np.ndarray) -> Any:
        """Standard action -> benchmark native action format.

        Default: takes first 6 dimensions (many benchmarks lack gripper).
        Subclasses can override if needed.
        """
        return action[:6]
