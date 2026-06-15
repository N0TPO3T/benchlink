#!/usr/bin/env python3
"""
LIBERO → BenchmarkRunner.

LIBERO (LIbrary for Embodied Robot OOD):
  - 100 simulation manipulation tasks (LIBERO-100) / 10 tasks (LIBERO-10)
  - Uses OffScreenRenderEnv (headless rendering, suitable for batch evaluation)
  - Observations: agentview RGB, wrist RGB, joint pos, gripper state
  - Actions: 7-D joint position target or delta joint

Data mapping:
  agentview_image (256,256,3)    → rgb_static
  robot0_eye_in_hand_image (128,128,3) → rgb_gripper
  robot0_joint_pos (7,)          → proprio
  task_name / language           → language

Action mapping:
  Standard 7-D delta_pose → LIBERO 7-D joint target
  Uses simple incremental strategy: joint_target = current_joints + [delta_pose[:6], gripper_delta]

Usage:
    runner = LiberoRunner()
    runner.setup({"task_name": "open_the_middle_drawer"})
    results = runner.evaluate(model, n_episodes=10)
    # → {"success_rate": 0.8, "n_episodes": 10, ...}
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs


class LiberoRunner(BenchmarkRunner):
    """LIBERO simulation evaluation executor."""

    def __init__(self):
        super().__init__()
        self.env = None
        self.task_name: str = ""
        self.task_description: str = ""
        self._max_steps: int = 200
        self._has_tactile: bool = False

    def setup(self, config: dict) -> None:
        """Initialize LIBERO simulation environment.

        Required config fields:
            task_name:     task name (e.g., "open_the_middle_drawer")
            task_root:     LIBERO task_embedding / bddl root directory
            task_suite:    task suite name (e.g., "libero_spatial", "libero_object")
            task_id:       task id (default 0)
        Optional config fields:
            max_steps:     max steps per episode (default 200)
            resolution:    render resolution (default 256)
            seed:          random seed (default 42)
            libero_root:   LIBERO repo path (added to sys.path)
        """
        self.task_name = config["task_name"]
        self._max_steps = config.get("max_steps", 200)
        seed = config.get("seed", 42)
        task_suite = config.get("task_suite", "libero_spatial")
        task_id = config.get("task_id", 0)

        # Add LIBERO repo to Python path
        libero_root = config.get("libero_root")
        if libero_root:
            import sys
            if str(libero_root) not in sys.path:
                sys.path.insert(0, str(libero_root))

        # ── Lazy import LIBERO (avoid I/O overhead of get_libero_path) ──
        from libero.libero.envs import OffScreenRenderEnv

        # Construct task embedding / BDDL path
        task_root = Path(config["task_root"])
        task_embed_path = str(task_root / self.task_name)

        self.env = OffScreenRenderEnv(
            task_embedding_path=task_embed_path,
            task_name=self.task_name,
            resolution=config.get("resolution", 256),
        )
        self.task_description = self.env.task_description

        # Set random seed (some LIBERO versions do not support seed() method)
        try:
            self.env.seed(seed)
        except (AttributeError, TypeError):
            pass

        self.config = config

    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run N episodes of simulation evaluation.

        Args:
            model:      loaded ModelAdapter
            n_episodes: number of evaluation episodes

        Returns:
            dict: {
                "success_rate": float,
                "success_rate_std": float,
                "n_episodes": int,
                "mean_episode_length": float,
                "total_steps": int,
            }
        """
        successes = []
        episode_lengths = []

        for ep in range(n_episodes):
            # ── Environment reset ──
            raw_obs, _ = self.env.reset()
            model.reset()
            done = False
            step = 0

            # ── Interaction loop ──
            for step in range(self._max_steps):
                canonical = self._to_canonical(raw_obs)

                # Check tactile fields and fill placeholders
                if not self._has_tactile:
                    canonical.tactile_img = None
                    canonical.tactile_feat = None

                # Model inference
                action = model.act(canonical)  # (7,)

                # Convert to native environment action → step
                env_action = self._from_canonical(action)
                raw_obs, reward, done, trunc, info = self.env.step(env_action)

                if done or trunc:
                    break

            # ── Record this episode results ──
            episode_lengths.append(step + 1)
            successes.append(1.0 if info.get("success", False) or done else 0.0)

        succ_arr = np.array(successes)

        return {
            "success_rate": float(np.mean(succ_arr)),
            "success_rate_std": float(np.std(succ_arr)),
            "n_episodes": n_episodes,
            "mean_episode_length": float(np.mean(episode_lengths)),
            "total_steps": int(np.sum(episode_lengths)),
        }

    # ── Format conversion ──

    def _to_canonical(self, raw_obs: dict) -> CanonicalObs:
        """LIBERO native observation → CanonicalObs."""
        return CanonicalObs(
            rgb_static=raw_obs.get("agentview_image"),
            rgb_gripper=raw_obs.get("robot0_eye_in_hand_image"),
            depth=None,
            proprio=raw_obs.get("robot0_joint_pos"),   # (7,)
            tactile_img=None,
            tactile_feat=None,
            language=self.task_description,
            lang_embed=self.env.get_task_embedding() if hasattr(self.env, "get_task_embedding") else None,
        )

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → LIBERO native action (7,) joint target.

        Standard: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        LIBERO:   [joint_0..joint_6] — joint position target

        Conversion strategy: simple increment (suitable for small-step delta)
        """
        # Get current joint positions
        current_joints = self.env._env.get_robot0_joint_positions()
        joint_target = current_joints.copy()

        # Add first 6-dim delta to joints
        # Assume first 3 joints correspond to xyz translation, last 3 to rotation
        joint_target[:6] += action[:6]

        # gripper (7th dim) directly mapped to finger joint
        joint_target[6] = action[6]

        return joint_target

    def close(self) -> None:
        """Clean up environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
