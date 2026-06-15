#!/usr/bin/env python3
"""
RoboTwin → BenchmarkRunner.

RoboTwin 2.0:
  - Dual-arm robot teleoperation dataset
  - Contains multi-view RGB images, joint states, action sequences
  - Offline replay evaluation: sample trajectory segments, compare model predictions with GT actions

Data format (zarr):
  meta/episode_ends:   (N,) — end index for each episode
  data/action:         (T, D) — action (D=7: 6-D delta + gripper)
  data/state:          (T, D) — robot state (joint pos or ee pose)
  data/front:          (T, H, W, 3) — front view images
  data/wrist:          (T, H, W, 3) — wrist view images
  data/hand:           (T, H, W, 3) — hand view images (some versions)

Action mapping:
  RoboTwin native: [dx, dy, dz, drot_x, drot_y, drot_z, gripper] (7,)
  Standard action: [dx, dy, dz, droll, dpitch, dyaw, gripper]  — passthrough

Dependencies:
  pip install zarr

Usage:
    runner = RoboTwinRunner()
    runner.setup({"data_path": "/path/to/robotwin/data.zarr"})
    results = runner.evaluate(model, n_episodes=5)
    # → {"position_error_mean": 0.012, "rotation_error_mean": 0.08, ...}
"""

import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class RoboTwinRunner(BenchmarkRunner):
    """RoboTwin offline replay evaluator."""

    def __init__(self):
        super().__init__()
        self.data_root: Optional[Path] = None
        self._zarr = None
        self._episode_ranges: list = []
        self._n_episodes: int = 0
        self._horizon: int = 50
        self.task_description: str = ""

    def setup(self, config: dict) -> None:
        """Initialize RoboTwin evaluation configuration.

        Required config fields:
            data_path: zarr dataset path
        Optional config fields:
            horizon:          number of frames per sampled segment (default 50)
            seed:             random seed (default 42)
            task_description: task description text
            img_keys:         image key mapping (for adapting different RoboTwin versions)
        """
        import zarr

        self.data_root = Path(config["data_path"])
        if not self.data_root.exists():
            raise FileNotFoundError(f"RoboTwin data not found: {self.data_root}")

        self._zarr = zarr.open(str(self.data_root), mode="r")
        self._horizon = config.get("horizon", 50)
        self.task_description = config.get("task_description", "robot twin manipulation task")

        # Read episode boundaries
        episode_ends = self._zarr["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])
        self._episode_ranges = list(zip(starts.tolist(), episode_ends.tolist()))
        self._n_episodes = len(self._episode_ranges)

        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        # Image key mapping (adapt for different RoboTwin versions)
        self._img_keys = config.get("img_keys", {
            "rgb_static": "front",
            "rgb_gripper": "wrist",
        })

        self.config = config

    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run N episodes of offline replay evaluation.

        Args:
            model:      loaded ModelAdapter
            n_episodes: number of episodes to sample

        Returns:
            dict: {
                "position_error_mean": float,
                "position_error_std": float,
                "rotation_error_mean": float,
                "rotation_error_std": float,
                "position_error_median": float,
                "rotation_error_median": float,
                "n_timesteps": int,
                "n_episodes": int,
            }
        """
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

                # GT action
                gt_action = self._get_gt_action(t)  # (7,)

                # Error (only the first 6 dims of delta pose)
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
        """zarr frame t → CanonicalObs.

        Handles multiple RoboTwin data versions:
          - original: front / wrist / hand
          - openxembodiment: rgb_static / rgb_gripper
        """
        z = self._zarr["data"]

        # Try multiple keys (different RoboTwin versions use different key names)
        rgb_static = self._try_get(z, ["front", "rgb_static", "image", "rgb"], t)
        rgb_gripper = self._try_get(z, ["wrist", "hand", "rgb_gripper", "wrist_rgb"], t)
        proprio = self._try_get(z, ["state", "joint_positions", "proprio", "robot_state"], t)

        return CanonicalObs(
            rgb_static=rgb_static,
            rgb_gripper=rgb_gripper,
            depth=None,
            proprio=proprio,
            tactile_img=None,
            tactile_feat=None,
            language=self.task_description,
            lang_embed=None,
        )

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → RoboTwin native action (7,).

        RoboTwin action format matches the standard format:
          [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        return action.copy()

    def _get_gt_action(self, t: int) -> np.ndarray:
        """Get GT action at frame t, convert to standard format (STANDARD_ACTION_DIM,)."""
        raw = self._zarr["data/action"][t]
        arr = np.asarray(raw, dtype=np.float64)

        if arr.shape[0] >= STANDARD_ACTION_DIM:
            return arr[:STANDARD_ACTION_DIM]
        elif arr.shape[0] == 6:
            # 6-D action (no gripper)
            return np.concatenate([arr, [0.0]])
        else:
            # Low-dimensional action, pad to standard size
            return np.pad(arr, (0, STANDARD_ACTION_DIM - arr.shape[0]), constant_values=0.0)

    @staticmethod
    def _try_get(zarr_group: Any, keys: list, t: int) -> Optional[np.ndarray]:
        """Try reading frame t from a zarr group using multiple keys."""
        for k in keys:
            if k in zarr_group:
                data = zarr_group[k][t]
                return np.asarray(data)
        return None

    def close(self) -> None:
        """Clean up zarr store."""
        if self._zarr is not None:
            self._zarr.store.close()
            self._zarr = None
