#!/usr/bin/env python3
"""
DreamZero -> ModelAdapter.

Inherits from DockerModelAdapter base class.

DreamZero (World Action Model):
  - 16.5B world model + actor (INT8 quantized)
  - Input: RGB image + language instruction + history latent
  - Output: latent action -> VAE decode -> 7-D continuous action

Docker image: dreamzero:int8 / dreamzero:int8-final

Usage:
    adapter = DreamZeroAdapter()
    adapter.load("", {"container_name": "dreamzero_server"})
    action = adapter.act(canonical_obs)  # -> (7,)
"""

import numpy as np

from benchlink.models.docker_base import DockerModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class DreamZeroAdapter(DockerModelAdapter):
    """DreamZero adapter -- Docker exec JSON line protocol."""

    def _get_defaults(self) -> dict:
        return {
            "container_name": "dreamzero_server",
            "image": "dreamzero:int8-final",
            "server_script": "/workspace/dreamzero_inference_server.py",
            "server_start_timeout": 180,  # 16.5B model loads slowly
            "conda_env": "",
            "action_low": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0],
            "action_high": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        }

    def _build_request(self, obs: CanonicalObs) -> dict:
        """Construct DreamZero request from CanonicalObs."""
        request = {
            "obs": {
                "rgb_static": obs.rgb_static.tolist(),
                "language": obs.language or "",
            },
            "id": self._req_id,
        }
        if obs.proprio is not None:
            request["obs"]["proprio"] = obs.proprio.tolist()
        if obs.tactile_img is not None:
            request["obs"]["tactile_img"] = obs.tactile_img.tolist()
        return request

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,)."""
        if not self._server_ready:
            raise RuntimeError("DreamZeroAdapter server not ready. Call load() first.")
        if obs.rgb_static is None:
            raise ValueError("DreamZeroAdapter requires obs.rgb_static")

        self._req_id += 1
        request = self._build_request(obs)
        action = self._send_request(request)
        return self._clip_action(action).astype(np.float64)
