#!/usr/bin/env python3
"""
ManiSkill → BenchmarkRunner。

ManiSkill (v2/v3):
  - GPU 加速仿真环境（SAPIEN + MuJoCo 后端）
  - 多种控制模式: pd_joint_delta_pos, pd_ee_delta_pos, etc.
  - 观察: rgb, depth, 关节位置, 夹爪状态
  - 动作: 取决于 control_mode，默认 7-D joint delta + gripper

数据映射:
  image/rgb (H,W,3)             → rgb_static
  image/depth (H,W)             → depth
  robot_joint_pos (7,)          → proprio
  gripper_qpos (2,)             → extra["gripper_qpos"]
  task_description              → language

动作映射:
  标准 7-D delta_pose + gripper → ManiSkill 原生格式
  裁剪到 env.action_space 边界

依赖:
  pip install mani_skill (v3) 或 mani-skill2 (v2)
  pip install gymnasium

用法:
    runner = ManiSkillRunner()
    runner.setup({"env_id": "PickCube-v1", "control_mode": "pd_joint_delta_pos"})
    results = runner.evaluate(model, n_episodes=10)
    # → {"success_rate": 0.8, ...}
"""

import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class ManiSkillRunner(BenchmarkRunner):
    """ManiSkill 仿真评测执行器。"""

    def __init__(self):
        super().__init__()
        self.env = None
        self.task_description: str = ""
        self._max_steps: int = 200
        self._action_dim: int = 7
        self._action_low: Optional[np.ndarray] = None
        self._action_high: Optional[np.ndarray] = None

    def setup(self, config: dict) -> None:
        """初始化 ManiSkill 仿真环境。

        config 必需字段:
            env_id:     环境 ID (如 "PickCube-v1", "StackCube-v1")
        config 可选字段:
            obs_mode:       观测模式 (默认 "state_dict", 可选 "rgb", "rgbd")
            control_mode:   控制模式 (默认 "pd_joint_delta_pos")
            sim_backend:    仿真后端 (默认 "gpu", 可选 "cpu")
            max_steps:      每 episode 最大步数 (默认 200)
            task_description: 任务描述文本（用于 _to_canonical 的 language）
            seed:           随机种子 (默认 42)
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
        import mani_skill.envs  # noqa: F401 — 注册 envs

        self.env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend=sim_backend,
        )

        # ── 缓存 action space 边界（用于动作裁剪） ──
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
        """跑 N 个 episode 的仿真评测。

        Args:
            model:      已加载的 ModelAdapter
            n_episodes: 评测 episode 数

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
            # ManiSkill v3 用 "success", v2 用 "is_success"
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

    # ── 格式转换 ──

    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        """ManiSkill 原生观察 → CanonicalObs。

        支持 state_dict 模式（推荐）和原始 dict 模式。
        """
        rgb = None
        depth = None
        proprio = None

        if isinstance(raw_obs, dict):
            # ── state_dict 模式: obs["image"]={"rgb": ..., "depth": ...} ──
            img_dict = raw_obs.get("image", {})
            if isinstance(img_dict, dict):
                rgb = img_dict.get("rgb", rgb)
                depth = img_dict.get("depth", depth)
            elif isinstance(img_dict, np.ndarray):
                rgb = img_dict

            # ── 多相机模式: 取第一台相机的 RGB ──
            if rgb is None:
                for cam_k in ("camera0", "cam0", "hand_camera", "base_camera"):
                    cam = raw_obs.get(cam_k, {})
                    if isinstance(cam, dict):
                        rgb = cam.get("rgb", rgb)
                        if rgb is not None:
                            break

            # ── proprio: 机器人状态（用 None 安全链替代 or, 避免 numpy 数组触发 ValueError） ──
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
        """标准动作 (7,) → ManiSkill 原生动作。

        裁剪到 action_space 边界以防 env 报错。
        """
        env_action = action.copy().astype(np.float64)

        # 适配 action 空间维度
        if self._action_dim < STANDARD_ACTION_DIM:
            env_action = env_action[:self._action_dim]
        elif self._action_dim > STANDARD_ACTION_DIM:
            env_action = np.pad(env_action, (0, self._action_dim - STANDARD_ACTION_DIM))

        # 裁剪边界
        if self._action_low is not None:
            env_action = np.clip(
                env_action,
                self._action_low[:len(env_action)],
                self._action_high[:len(env_action)],
            )

        return env_action

    def close(self) -> None:
        """清理环境。"""
        if self.env is not None:
            self.env.close()
            self.env = None
