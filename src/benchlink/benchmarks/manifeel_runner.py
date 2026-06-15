#!/usr/bin/env python3
"""
ManiFeelRunner — offline replay evaluation for USB insertion/removal assembly tasks.

P0 mode: offline replay (no simulation environment involved)
  - Sample continuous trajectory segments from zarr dataset
  - Feed observations to the Adapter each frame, model outputs action
  - Compare against ground truth actions (position/rotation error)
  - Does not execute simulation rollback (P1 will do online evaluation)

Data format (zarr):
  Keys:
    data/
      action:                     (5976, 6) float32 — [dx, dy, dz, drot_x, drot_y, drot_z]
      state:                      (5976, 7) float32 — [x, y, z, qw, qx, qy, qz]
      front:                      (5976, 256, 256, 3)
      wrist / wrist_2:            (5976, 256, 256, 3)
      side:                       (5976, 256, 256, 3)
      left_tactile_camera_taxim:  (5976, 320, 240, 3)
      right_tactile_camera_taxim: (5976, 320, 240, 3)
      tactile_depth_right:        (5976, 10, 14)
      tactile_force_field_right:  (5976, 10, 14, 3)
    meta/
      episode_ends:               (50,) — end index for each episode

Action mapping:
  ManiFeel native: [dx, dy, dz, drot_x, drot_y, drot_z] (axis-angle delta)
  Standard action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  Conversion:     native + pad gripper=0 = standard;  standard[:6] = native
"""

import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs


class ManiFeelRunner(BenchmarkRunner):
    """ManiFeel offline replay evaluator."""

    def __init__(self):
        super().__init__()
        self.data_root: Optional[Path] = None
        self.img_transform = None
        self._episode_ranges: list = []   # [(start, end), ...]
        self._n_episodes: int = 50
        self._horizon: int = 50

        # Lazy loading (open zarr only once)
        self._zarr = None

    def setup(self, config: dict) -> None:
        """Initialize ManiFeel evaluation configuration.

        Required config fields:
            data_path: zarr dataset path
        Optional config fields:
            horizon:  number of frames per sampled segment (default 50)
            seed:     random seed (default 42)
            mode:     "offline" | "online" (P0 uses offline only)
        """
        self.data_root = Path(config["data_path"])
        if not self.data_root.exists():
            raise FileNotFoundError(f"ManiFeel data not found: {self.data_root}")

        self._horizon = config.get("horizon", 50)
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        # Read episode boundaries
        import zarr
        self._zarr = zarr.open(str(self.data_root), mode="r")
        episode_ends = self._zarr["meta/episode_ends"][:]  # (50,)
        starts = np.concatenate([[0], episode_ends[:-1]])
        self._episode_ranges = list(zip(starts.tolist(), episode_ends.tolist()))
        self._n_episodes = len(self._episode_ranges)

        # Image preprocessing (configurable)
        self.img_transform = config.get(
            "img_transform",
            lambda x: x,  # P0: raw image, no resize
        )

        self.config = config

    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run N episodes of offline replay evaluation.

        Args:
            model:     loaded ModelAdapter
            n_episodes: number of episodes to sample (default 5)

        Returns:
            dict: {
                "position_error_mean": float,
                "position_error_std":  float,
                "rotation_error_mean": float,
                "rotation_error_std":  float,
                "position_error_median": float,
                "rotation_error_median": float,
                "n_timesteps": int,
                "n_episodes": int,
            }
        """
        # ── Sample n_episodes trajectories ──
        n = min(n_episodes, self._n_episodes)
        selected = random.sample(self._episode_ranges, n)

        pos_errors = []
        rot_errors = []

        for ep_idx, (start, end) in enumerate(selected):
            length = end - start
            if length < self._horizon:
                continue  # Skip episodes that are too short

            # Randomly choose a start point within the episode
            t_start = start + random.randint(0, length - self._horizon)
            t_end = t_start + self._horizon

            model.reset()
            for t in range(t_start, t_end):
                # Construct standard observation
                obs = self._to_canonical(t)

                # Model inference
                pred_action = model.act(obs)  # (7,)

                # Get GT action (standard format)
                gt_action = self._get_gt_action(t)  # (7,)

                # Compute error (only the first 6 dims of delta pose)
                pos_err = np.linalg.norm(pred_action[:3] - gt_action[:3])
                rot_err = np.linalg.norm(pred_action[3:6] - gt_action[3:6])

                pos_errors.append(pos_err)
                rot_errors.append(rot_err)

        if not pos_errors:
            return {
                "error": "No valid episodes found (all too short)",
                "n_episodes": 0,
                "n_timesteps": 0,
            }

        pos_arr = np.array(pos_errors)
        rot_arr = np.array(rot_errors)

        return {
            "position_error_mean":   float(np.mean(pos_arr)),
            "position_error_std":    float(np.std(pos_arr)),
            "position_error_median": float(np.median(pos_arr)),
            "rotation_error_mean":   float(np.mean(rot_arr)),
            "rotation_error_std":    float(np.std(rot_arr)),
            "rotation_error_median": float(np.median(rot_arr)),
            "n_timesteps":           len(pos_errors),
            "n_episodes":            n,
        }

    def _to_canonical(self, t: int) -> CanonicalObs:
        """zarr frame t → CanonicalObs."""
        z = self._zarr["data"]

        return CanonicalObs(
            rgb_static=   self.img_transform(z["front"][t]),
            rgb_gripper=  self.img_transform(z["wrist"][t]),
            depth=None,
            proprio=      z["state"][t],           # (7,): xyz + quaternion
            tactile_img=  z["left_tactile_camera_taxim"][t],  # (320,240,3)
            tactile_feat=None,           # P0: left for the Adapter to decide
            tactile_depth=z["tactile_depth_right"][t],        # (10,14)
            tactile_force=z["tactile_force_field_right"][t],  # (10,14,3)
            language="insert the USB connector into the port",
            lang_embed=None,
        )

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → ManiFeel native action (6,).

        ManiFeel native action has no gripper and rotation format is axis-angle delta.
        The standard droll/dpitch/dyaw and ManiFeel's drot_x/drot_y/drot_z
        are equivalent for small-angle deltas.

        Returns:
            ndarray shape=(6,): [dx, dy, dz, drot_x, drot_y, drot_z]
        """
        return action[:6].copy()

    def _get_gt_action(self, t: int) -> np.ndarray:
        """Get the GT action at frame t, converted to standard format (7,).

        ManiFeel action: [dx, dy, dz, drot_x, drot_y, drot_z] (6,)
        Standard action: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7,)
        """
        raw = self._zarr["data/action"][t]  # (6,)
        return np.concatenate([raw, [0.0]])  # gripper=0
