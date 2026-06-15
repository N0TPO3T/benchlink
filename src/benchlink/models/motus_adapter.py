#!/usr/bin/env python3
"""
Motus -> ModelAdapter.

Inherits from DockerModelAdapter base class.
Motus uses image sequence + action chunk cache.

Docker image: motus:full

Usage:
    adapter = MotusAdapter()
    adapter.load("", {"container_name": "motus_server"})
    action = adapter.act(canonical_obs)  # -> (7,)
"""

from collections import deque
from typing import Optional

import numpy as np

from benchlink.models.docker_base import DockerModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class MotusAdapter(DockerModelAdapter):
    """Motus adapter -- Docker persistent inference + image sequence buffer."""

    def __init__(self):
        super().__init__()
        # Image sequence configuration
        self.n_obs_steps: int = 2
        self.n_action_steps: int = 8
        self.obs_history: deque = deque(maxlen=10)
        self._cached_action_chunk: Optional[np.ndarray] = None
        self._action_counter: int = 0

    def _get_defaults(self) -> dict:
        return {
            "container_name": "motus_server",
            "image": "motus:full",
            "server_script": "/workspace/motus_inference_server.py",
            "server_start_timeout": 120,
            "action_low": [-0.5, -0.5, -0.5, -1.0, -1.0, -1.0, 0.0],
            "action_high": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        }

    def load(self, checkpoint: str, config: dict) -> None:
        """Load Motus-specific configuration parameters."""
        self.n_obs_steps = config.get("n_obs_steps", 2)
        self.n_action_steps = config.get("n_action_steps", 8)
        self.obs_history = deque(maxlen=self.n_obs_steps)
        self._action_counter = 0
        self._cached_action_chunk = None
        super().load(checkpoint, config)

    def _build_request(self, obs: CanonicalObs) -> dict:
        """Build single-frame request (actual multi-frame request in _predict_chunk)."""
        request = {
            "obs": {
                "rgb_static": obs.rgb_static.tolist(),
                "language": obs.language or "",
            },
            "id": self._req_id,
        }
        if obs.proprio is not None:
            request["obs"]["proprio"] = obs.proprio.tolist()
        return request

    def _build_chunk_request(self) -> dict:
        """Build multi-frame request from history."""
        self._req_id += 1
        request = {
            "obs": {
                "rgb_static": [h.rgb_static.tolist() for h in self.obs_history],
                "language": self.obs_history[-1].language or "",
            },
            "id": self._req_id,
        }
        # Tactile images (if available)
        tactiles = [h.tactile_img for h in self.obs_history if h.tactile_img is not None]
        if tactiles:
            request["obs"]["tactile_img"] = [t.tolist() for t in tactiles]
        # Proprio (most recent frame)
        proprios = [h.proprio for h in self.obs_history if h.proprio is not None]
        if proprios:
            request["obs"]["proprio"] = proprios[-1].tolist()
        return request

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, uses action chunk cache scheduling."""
        if not self._server_ready:
            raise RuntimeError("MotusAdapter server not ready. Call load() first.")
        if obs.rgb_static is None:
            raise ValueError("MotusAdapter requires obs.rgb_static")

        # Save current frame to history
        self.obs_history.append(obs)

        # Action chunk cache scheduling
        if self._cached_action_chunk is None or self._action_counter >= self.n_action_steps:
            if len(self.obs_history) < self.n_obs_steps:
                return np.zeros(STANDARD_ACTION_DIM, dtype=np.float64)

            action_chunk = self._predict_chunk()
            self._cached_action_chunk = action_chunk[:self.n_action_steps]
            self._action_counter = 0

        step_action = self._cached_action_chunk[self._action_counter]
        self._action_counter += 1
        return self._clip_action(np.asarray(step_action, dtype=np.float64).flatten())

    def _predict_chunk(self) -> np.ndarray:
        """Request action chunk prediction from server using history frames."""
        request = self._build_chunk_request()
        raw = self._send_request(request)
        return np.asarray(raw, dtype=np.float64)

    def reset(self) -> None:
        """Reset rollout history + action cache."""
        self.obs_history.clear()
        self._action_counter = 0
        self._cached_action_chunk = None
        super().reset()
