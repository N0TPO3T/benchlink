#!/usr/bin/env python3
"""
RoboTwin → BenchmarkRunner。

RoboTwin 2.0:
  - 双臂机器人遥操作数据集
  - 包含多视角 RGB 图像、关节状态、动作序列
  - 离线 replay 评测：采样轨迹片段，对比模型预测与 GT 动作

数据格式 (zarr):
  meta/episode_ends:   (N,) — 每个 episode 的结束索引
  data/action:         (T, D) — 动作 (D=7: 6-D delta + gripper)
  data/state:          (T, D) — 机器人状态 (joint pos 或 ee pose)
  data/front:          (T, H, W, 3) — 正面视角图像
  data/wrist:          (T, H, W, 3) — 腕部视角图像
  data/hand:           (T, H, W, 3) — 手部视角图像（部分版本）

动作映射:
  RoboTwin 原生: [dx, dy, dz, drot_x, drot_y, drot_z, gripper] (7,)
  标准动作:     [dx, dy, dz, droll, dpitch, dyaw, gripper]  — 直通

依赖:
  pip install zarr

用法:
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
    """RoboTwin 离线 replay 评测器。"""

    def __init__(self):
        super().__init__()
        self.data_root: Optional[Path] = None
        self._zarr = None
        self._episode_ranges: list = []
        self._n_episodes: int = 0
        self._horizon: int = 50
        self.task_description: str = ""

    def setup(self, config: dict) -> None:
        """初始化 RoboTwin 评测配置。

        config 必需字段:
            data_path: zarr 数据集路径
        config 可选字段:
            horizon:          每段采样帧数 (默认 50)
            seed:             随机种子 (默认 42)
            task_description: 任务描述文本
            img_keys:         图像 key 映射 (用于适配不同 RoboTwin 版本)
        """
        import zarr

        self.data_root = Path(config["data_path"])
        if not self.data_root.exists():
            raise FileNotFoundError(f"RoboTwin data not found: {self.data_root}")

        self._zarr = zarr.open(str(self.data_root), mode="r")
        self._horizon = config.get("horizon", 50)
        self.task_description = config.get("task_description", "robot twin manipulation task")

        # 读取 episode 边界
        episode_ends = self._zarr["meta/episode_ends"][:]
        starts = np.concatenate([[0], episode_ends[:-1]])
        self._episode_ranges = list(zip(starts.tolist(), episode_ends.tolist()))
        self._n_episodes = len(self._episode_ranges)

        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        # 图像 key 映射（适配不同 RoboTwin 版本）
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
        """跑 N 个 episode 的离线 replay 评测。

        Args:
            model:      已加载的 ModelAdapter
            n_episodes: 采样的 episode 数

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
                continue  # 跳过过短的 episode

            # 在 episode 内随机取起点
            t_start = start + random.randint(0, length - self._horizon)
            t_end = t_start + self._horizon

            model.reset()
            for t in range(t_start, t_end):
                # 构造标准观察
                obs = self._to_canonical(t)

                # 模型推理
                pred_action = model.act(obs)  # (7,)

                # GT 动作
                gt_action = self._get_gt_action(t)  # (7,)

                # 误差（只看前 6 维 delta pose）
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
        """zarr 第 t 帧 → CanonicalObs。

        处理多种 RoboTwin 数据版本:
          - original: front / wrist / hand
          - openxembodiment: rgb_static / rgb_gripper
        """
        z = self._zarr["data"]

        # 多 key 尝试（不同 RoboTwin 版本 key 名不同）
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
        """标准动作 (7,) → RoboTwin 原生动作 (7,)。

        RoboTwin 动作格式与标准格式一致:
          [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        return action.copy()

    def _get_gt_action(self, t: int) -> np.ndarray:
        """取第 t 帧的 GT 动作，转为标准格式 (STANDARD_ACTION_DIM,)。"""
        raw = self._zarr["data/action"][t]
        arr = np.asarray(raw, dtype=np.float64)

        if arr.shape[0] >= STANDARD_ACTION_DIM:
            return arr[:STANDARD_ACTION_DIM]
        elif arr.shape[0] == 6:
            # 6-D action (no gripper)
            return np.concatenate([arr, [0.0]])
        else:
            # 低维 action，补齐
            return np.pad(arr, (0, STANDARD_ACTION_DIM - arr.shape[0]), constant_values=0.0)

    @staticmethod
    def _try_get(zarr_group: Any, keys: list, t: int) -> Optional[np.ndarray]:
        """在 zarr group 中按多个 key 尝试读取第 t 帧。"""
        for k in keys:
            if k in zarr_group:
                data = zarr_group[k][t]
                return np.asarray(data)
        return None

    def close(self) -> None:
        """清理 zarr store。"""
        if self._zarr is not None:
            self._zarr.store.close()
            self._zarr = None
