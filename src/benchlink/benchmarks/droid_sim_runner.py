#!/usr/bin/env python3
"""
DroidSim → BenchmarkRunner.

DROID (Distributed Robot Interaction Dataset):
  - Large-scale in-the-wild robotic manipulation dataset
  - sim_evals provides a MuJoCo-based simulation evaluation suite
  - Observations: multi-view RGB-D, joint positions, force/torque
  - Actions: 6-D delta pose (no gripper) or 7-D

Data mapping:
  image_front (H,W,3)           → rgb_static
  image_wrist (H,W,3)           → rgb_gripper
  depth_front (H,W)             → depth
  joint_positions (N,)          → proprio
  task_description              → language

Action mapping:
  Standard 7-D delta_pose + gripper → DROID native format
  Most DROID tasks use 6-D delta (no gripper)

Dependencies:
  pip install sim_evals or droid_sim

Usage:
    runner = DroidSimRunner()
    runner.setup({"task_name": "droid_pick_up_cup"})
    results = runner.evaluate(model, n_episodes=10)
    # → {"success_rate": 0.7, ...}
"""

import random
from typing import Dict, Any

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs


class DroidSimRunner(BenchmarkRunner):
    """DroidSim simulation evaluation executor."""

    def __init__(self):
        super().__init__()
        self.env = None
        self.task_name: str = ""
        self.task_description: str = ""
        self._max_steps: int = 300
        self._seed: int = 42

    def setup(self, config: dict) -> None:
        """Initialize DROID simulation environment.

        Required config fields:
            task_name: task name (e.g., "droid_pick_up_cup", "droid_open_drawer")
        Optional config fields:
            max_steps:      max steps per episode (default 300)
            task_description: task description text
            obs_type:       observation type (default "pixels", optional "state")
            seed:           random seed (default 42)
            sim_lib:        simulation library name (default "sim_evals", optional "droid_sim")
        """
        self.task_name = config.get("task_name", "")
        self._max_steps = config.get("max_steps", 300)
        self.task_description = config.get("task_description", "")
        self._seed = config.get("seed", 42)
        random.seed(self._seed)
        np.random.seed(self._seed)
        sim_lib = config.get("sim_lib", "sim_evals")

        # ── Lazy import (DROID simulation library may not be installed) ──
        if sim_lib == "sim_evals":
            from sim_evals.droid_env import DroidEnv
        elif sim_lib == "droid_sim":
            from droid_sim import DroidEnv as _DroidEnv
            DroidEnv = _DroidEnv
        else:
            raise ValueError(f"Unknown sim_lib: {sim_lib}")

        self.env = DroidEnv(
            task=self.task_name,
            obs_type=config.get("obs_type", "pixels"),
            seed=self._seed,
        )

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
            raw_obs = self.env.reset(seed=self._seed + ep)
            model.reset()
            step = 0

            for step in range(self._max_steps):
                canonical = self._to_canonical(raw_obs)
                action = model.act(canonical)  # (7,)
                env_action = self._from_canonical(action)

                # Compatible with Gymnasium 5-value (obs, rew, term, trunc, info) and legacy 4-value
                step_result = self.env.step(env_action)
                if len(step_result) == 5:
                    raw_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    raw_obs, reward, done, info = step_result

                if done:
                    break

            episode_lengths.append(step + 1)
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
        """DROID native observation → CanonicalObs.

        Compatible with multiple obs_type:
          - pixels (dict): {"image_front": ..., "image_wrist": ..., "joint_positions": ...}
          - state  (dict): {"joint_positions": ..., "gripper_state": ...}
        """
        if isinstance(raw_obs, dict):
            # Use None-safe chaining instead of 'or' to avoid ValueError from numpy arrays
            def _get_first(d: dict, *keys):
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        return v
                return None

            return CanonicalObs(
                rgb_static=_get_first(raw_obs, "image_front", "rgb_static"),
                rgb_gripper=_get_first(raw_obs, "image_wrist", "rgb_gripper"),
                depth=_get_first(raw_obs, "depth_front", "depth"),
                proprio=_get_first(
                    raw_obs, "joint_positions", "state", "proprio", "robot_state",
                ),
                language=self.task_description,
            )

        return CanonicalObs(language=self.task_description)

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → DROID native action.

        DROID commonly uses 6-D delta pose (position + axis-angle) + optional gripper.
        Default is to take only the first 6 dimensions.
        """
        # DROID action space is typically 6-D (no gripper)
        return action[:6].copy()

    def close(self) -> None:
        """Clean up environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
