#!/usr/bin/env python3
"""
ManiSkill → BenchmarkRunner.

ManiSkill (v2/v3):
  - GPU-accelerated simulation environment (SAPIEN + MuJoCo backend)
  - Multiple control modes: pd_joint_delta_pos, pd_ee_delta_pos, etc.
  - Observations: rgb, depth, joint positions, gripper state
  - Actions: depends on control_mode, default 7-D joint delta + gripper

Data mapping:
  image/rgb (H,W,3)             → rgb_static
  image/depth (H,W)             → depth
  robot_joint_pos (7,)          → proprio
  gripper_qpos (2,)             → extra["gripper_qpos"]
  task_description              → language

Action mapping:
  Standard 7-D delta_pose + gripper → ManiSkill native format
  Clipped to env.action_space bounds

Dependencies:
  pip install mani_skill (v3) or mani-skill2 (v2)
  pip install gymnasium

Usage:
    runner = ManiSkillRunner()
    runner.setup({"env_id": "PickCube-v1", "control_mode": "pd_joint_delta_pos"})
    results = runner.evaluate(model, n_episodes=10)
    # → {"success_rate": 0.8, ...}
"""

import random
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class ManiSkillRunner(BenchmarkRunner):
    """ManiSkill simulation evaluation executor."""

    def __init__(self):
        super().__init__()
        self.env = None
        self.task_description: str = ""
        self._max_steps: int = 200
        self._action_dim: int = 7
        self._action_low: Optional[np.ndarray] = None
        self._action_high: Optional[np.ndarray] = None

    def setup(self, config: dict) -> None:
        """Initialize ManiSkill simulation environment.

        Required config fields:
            env_id:     environment ID (e.g., "PickCube-v1", "StackCube-v1")
        Optional config fields:
            obs_mode:       observation mode (default "state_dict", optional "rgb", "rgbd")
            control_mode:   control mode (default "pd_joint_delta_pos")
            sim_backend:    simulation backend (default "gpu", optional "cpu")
            max_steps:      max steps per episode (default 200)
            task_description: task description text (used for _to_canonical language field)
            seed:           random seed (default 42)
        """
        env_id = config.get("env_id", "PickCube-v1")
        obs_mode = config.get("obs_mode", "state_dict")
        control_mode = config.get("control_mode", "pd_joint_delta_pos")
        sim_backend = config.get("sim_backend", "gpu")
        self._max_steps = config.get("max_steps", 200)
        self.task_description = config.get("task_description", "")
        ms_seed = config.get("seed", 42)
        random.seed(ms_seed)
        np.random.seed(ms_seed)

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 — register envs

        self.env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend=sim_backend,
        )

        # ── Cache action space bounds (for action clipping) ──
        if hasattr(self.env.action_space, "shape"):
            self._action_dim = self.env.action_space.shape[0]
        if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
            self._action_low = np.asarray(self.env.action_space.low, dtype=np.float64)
            self._action_high = np.asarray(self.env.action_space.high, dtype=np.float64)

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
        seed = kwargs.get("seed", self.config.get("seed", 42))

        for ep in range(n_episodes):
            raw_obs, _ = self.env.reset(seed=seed + ep)
            model.reset()
            step = 0

            for step in range(self._max_steps):
                canonical = self._to_canonical(raw_obs)
                action = model.act(canonical)  # (7,)
                env_action = self._from_canonical(action)
                raw_obs, reward, terminated, truncated, info = self.env.step(env_action)

                if terminated or truncated:
                    break

            episode_lengths.append(step + 1)
            # ManiSkill v3 uses "success", v2 uses "is_success"
            succ = info.get("success", info.get("is_success", False))
            successes.append(1.0 if succ else 0.0)

        succ_arr = np.array(successes)

        return {
            "success_rate": float(np.mean(succ_arr)),
            "success_rate_std": float(np.std(succ_arr)),
            "n_episodes": n_episodes,
            "mean_episode_length": float(np.mean(episode_lengths)),
            "total_steps": int(np.sum(episode_lengths)),
        }

    # ── Format conversion ──

    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        """ManiSkill native observation → CanonicalObs.

        Supports state_dict mode (recommended) and raw dict mode.
        """
        rgb = None
        depth = None
        proprio = None

        if isinstance(raw_obs, dict):
            # ── state_dict mode: obs["image"]={"rgb": ..., "depth": ...} ──
            img_dict = raw_obs.get("image", {})
            if isinstance(img_dict, dict):
                rgb = img_dict.get("rgb", rgb)
                depth = img_dict.get("depth", depth)
            elif isinstance(img_dict, np.ndarray):
                rgb = img_dict

            # ── Multi-camera mode: take RGB from the first camera ──
            if rgb is None:
                for cam_k in ("camera0", "cam0", "hand_camera", "base_camera"):
                    cam = raw_obs.get(cam_k, {})
                    if isinstance(cam, dict):
                        rgb = cam.get("rgb", rgb)
                        if rgb is not None:
                            break

            # ── proprio: robot state (use None-safe chaining instead of 'or' to avoid ValueError from numpy arrays) ──
            proprio = raw_obs.get("robot_joint_pos")
            if proprio is None:
                proprio = raw_obs.get("agent")
            if proprio is None:
                proprio = raw_obs.get("state")

        return CanonicalObs(
            rgb_static=rgb,
            depth=depth,
            proprio=proprio,
            language=self.task_description,
        )

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → ManiSkill native action.

        Clip to action_space bounds to prevent env errors.
        """
        env_action = action.copy().astype(np.float64)

        # Adapt to action space dimension
        if self._action_dim < STANDARD_ACTION_DIM:
            env_action = env_action[:self._action_dim]
        elif self._action_dim > STANDARD_ACTION_DIM:
            env_action = np.pad(env_action, (0, self._action_dim - STANDARD_ACTION_DIM))

        # Clip to bounds
        if self._action_low is not None:
            env_action = np.clip(
                env_action,
                self._action_low[:len(env_action)],
                self._action_high[:len(env_action)],
            )

        return env_action

    def close(self) -> None:
        """Clean up environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
