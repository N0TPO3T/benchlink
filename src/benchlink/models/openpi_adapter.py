#!/usr/bin/env python3
"""
OpenPI (pi0) -> ModelAdapter.

Inherits from DockerModelAdapter base class, only needs default config and request format.

Docker image: openpi:pi05-libero-rollout-ok

Usage:
    adapter = OpenPiAdapter()
    adapter.load("", {"container_name": "openpi_server"})
    action = adapter.act(canonical_obs)  # -> (7,)
"""

import numpy as np

from benchlink.models.docker_base import DockerModelAdapter
from benchlink.schema import CanonicalObs


class OpenPiAdapter(DockerModelAdapter):
    """OpenPI (pi0) adapter -- Docker persistent inference server mode."""

    def _get_defaults(self) -> dict:
        return {
            "container_name": "openpi_server",
            "image": "openpi:pi05-libero-rollout-ok",
            "server_script": "/workspace/openpi_inference_server.py",
            "server_start_timeout": 120,
            "action_low": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0],
            "action_high": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        }

    def _build_request(self, obs: CanonicalObs) -> dict:
        """Construct OpenPI request from CanonicalObs."""
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

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,)."""
        if not self._server_ready:
            raise RuntimeError("OpenPiAdapter server not ready. Call load() first.")
        if obs.rgb_static is None:
            raise ValueError("OpenPiAdapter requires obs.rgb_static")

        self._req_id += 1
        request = self._build_request(obs)
        action = self._send_request(request)
        return self._clip_action(action).astype(np.float64)
