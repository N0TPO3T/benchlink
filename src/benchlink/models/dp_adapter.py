#!/usr/bin/env python3
"""
DP (Diffusion Policy) -> ModelAdapter.

Diffusion Policy:
  - Standard diffusion policy, conditioned on multi-frame image history and robot state
  - Outputs action chunk (pred_horizon, action_dim)
  - Shares diffusion_policy backend library with RDP, but without tactile conditioning

Key difference from RDPAdapter:
  RDP: tactile_image + rgb_image + proprio -> action (tactile conditioning)
  DP:  rgb_image + proprio -> action (standard diffusion policy)

Usage:
    adapter = DPAdapter()
    adapter.load("/path/to/checkpoint.ckpt", {"device": "cuda"})
    action = adapter.act(canonical_obs)  # -> (7,)
"""

import sys
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np

from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class DPAdapter(ModelAdapter):
    """Diffusion Policy adapter -- standard action space output."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cpu"
        self.n_obs_steps: int = 2          # Input history frame count
        self.n_action_steps: int = 8        # Output action chunk length
        self.pred_horizon: int = 16         # Total prediction steps
        self.obs_history: Optional[deque] = None
        self.num_inference_steps: int = 10
        self._repo_root: Optional[Path] = None
        self._action_counter: int = 0       # Cumulative steps for action chunk scheduling

    def load(self, checkpoint: str, config: dict) -> None:
        """Load Diffusion Policy checkpoint.

        Args:
            checkpoint: Path to .ckpt file
            config: Config dict, supported fields:
                device:              Inference device (default "cuda")
                n_obs_steps:         Number of input frames (default 2)
                n_action_steps:      Action chunk length (default 8)
                pred_horizon:        Total prediction horizon (default 16)
                num_inference_steps: Inference steps (default 10)
                dp_repo:             diffusion_policy repo path (needed for server deployment)
        """
        self.device = config.get("device", "cuda")
        self.n_obs_steps = config.get("n_obs_steps", 2)
        self.n_action_steps = config.get("n_action_steps", 8)
        self.pred_horizon = config.get("pred_horizon", 16)
        self.num_inference_steps = config.get("num_inference_steps", 10)

        # Add diffusion_policy repo to sys.path
        repo_root = config.get("dp_repo", "")
        if repo_root:
            self._repo_root = Path(repo_root)
            repo_str = str(self._repo_root)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        import torch
        from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
            DiffusionUnetLowdimWorkspace,
        )

        # ── Load checkpoint ──
        payload = torch.load(checkpoint, map_location="cpu")
        cfg = payload.get("cfg", None)
        if cfg is None:
            cfg = {}

        self.model = DiffusionUnetLowdimWorkspace(cfg)
        self.model.load_payload(payload, exclude_keys=None, include_keys=None)
        self.model.to(self.device)
        self.model.eval()

        # ── Normalization stats (DP normalizes observations, denormalizes outputs) ──
        self._action_norm_stats = payload.get("action_normalizer", None) or \
                                  payload.get("ema_action_normalizer", None)
        self._obs_norm_stats = payload.get("obs_normalizer", None) or \
                               payload.get("ema_obs_normalizer", None)

        # ── Preprocessing pipeline (avoid re-building on every act()) ──
        import torchvision.transforms as T
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # ── Initialize history buffer ──
        self.obs_history = deque(maxlen=self.n_obs_steps)
        self._action_counter = 0
        self._cached_action_chunk = None

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,).

        Depends on obs fields:
            rgb_static: (H, W, 3) static camera image
            proprio:    (N,) robot state
        """
        if obs.rgb_static is None:
            raise ValueError("DPAdapter requires obs.rgb_static")
        if obs.proprio is None:
            raise ValueError("DPAdapter requires obs.proprio")

        import torch

        # ── Image preprocessing (using transform cached in load()) ──
        rgb_t = self._transform(obs.rgb_static).unsqueeze(0)  # (1, 3, 224, 224)
        proprio_t = torch.from_numpy(obs.proprio).float().unsqueeze(0)  # (1, N)

        # ── Observation normalization ──
        if self._obs_norm_stats is not None:
            # Build observation vector [rgb_flat; proprio]
            rgb_flat = rgb_t.flatten().unsqueeze(0)   # (1, 3*224*224)
            obs_vec = torch.cat([rgb_flat, proprio_t], dim=1)  # (1, obs_dim)
            mean_t = torch.as_tensor(self._obs_norm_stats["mean"], device=self.device)
            std_t  = torch.as_tensor(self._obs_norm_stats["std"], device=self.device)
            obs_vec = (obs_vec - mean_t) / std_t
            # Split back into rgb and proprio (restore original dimensions)
            rgb_flat_norm = obs_vec[:, :rgb_flat.shape[1]]
            proprio_norm  = obs_vec[:, rgb_flat.shape[1]:]
            # Reshape back to image
            rgb_t = rgb_flat_norm.reshape(1, 3, 224, 224)
            proprio_t = proprio_norm

        # ── Append to history ──
        self.obs_history.append((rgb_t.to(self.device), proprio_t.to(self.device)))

        # ── Check if re-inference is needed ──
        if self._cached_action_chunk is None or self._action_counter >= self.n_action_steps:
            if len(self.obs_history) < self.n_obs_steps:
                # Not enough history, output zero action
                return np.zeros(STANDARD_ACTION_DIM, dtype=np.float64)

            # ── Build model input ──
            # diffusion_policy's predict_action() expects obs_dict
            obs_dict = {
                "img": torch.stack([h[0] for h in self.obs_history], dim=1),   # (1, T, 3, H, W)
                "state": torch.stack([h[1] for h in self.obs_history], dim=1), # (1, T, state_dim)
            }

            with torch.no_grad():
                # Note: workspace.model is the actual policy, workspace itself has no predict()
                raw_action = self.model.model.predict_action(
                    obs_dict,
                    num_inference_steps=self.num_inference_steps,
                )  # (1, pred_horizon, action_dim)

            # ── Convert to numpy ──
            if isinstance(raw_action, torch.Tensor):
                raw_action = raw_action.cpu().numpy()
            action_chunk = raw_action[0] if raw_action.ndim == 3 else raw_action

            # ── Action denormalization ──
            if self._action_norm_stats is not None:
                mean_a = np.array(self._action_norm_stats.get("mean", 0), dtype=np.float64)
                std_a  = np.array(self._action_norm_stats.get("std", 1), dtype=np.float64)
                action_chunk = action_chunk * std_a + mean_a

            self._cached_action_chunk = action_chunk[:self.n_action_steps]
            self._action_counter = 0

        # ── Get step action from cache ──
        step_action = self._cached_action_chunk[self._action_counter]
        self._action_counter += 1

        # ── Convert to standard action format ──
        # diffusion_policy output: [action_dim],
        # first 6 dims are delta pose, 7th dim is gripper
        action = np.array(step_action, dtype=np.float64).flatten()
        if action.shape[0] < STANDARD_ACTION_DIM:
            # Pad to STANDARD_ACTION_DIM
            action = np.pad(action, (0, STANDARD_ACTION_DIM - action.shape[0]),
                            constant_values=0.0)
        elif action.shape[0] > STANDARD_ACTION_DIM:
            action = action[:STANDARD_ACTION_DIM]

        return action

    def reset(self) -> None:
        """Called at the start of each episode."""
        self.obs_history.clear()
        self._action_counter = 0
        self._cached_action_chunk = None
