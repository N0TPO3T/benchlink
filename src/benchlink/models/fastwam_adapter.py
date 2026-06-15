#!/usr/bin/env python3
"""
FastWAM -> ModelAdapter.

FastWAM (Fast World Action Model):
  - Lightweight WAM, inputs static camera image + language instruction
  - Outputs 50-step joint position action chunk (50, 7)
  - Native action space: absolute joint positions [joint_0..joint_6]
  - Standard action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]

Conversion logic:
  FastWAM outputs joint target -> difference from current joint -> delta pose
  Missing rotation filled with 0 (FastWAM does not output rotation)
"""

import numpy as np

from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class FastWAMAdapter(ModelAdapter):
    """FastWAM adapter -- direct import mode."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.action_horizon: int = 50

    def load(self, checkpoint: str, config: dict) -> None:
        """Load FastWAM model weights.

        Args:
            checkpoint: Path to FastWAM checkpoint (.ckpt)
            config: Config dict, supported fields:
                device:        Inference device (default "cuda")
                action_horizon: Action chunk length (default 50)
                img_size:       Input image size (default 224)
        """
        self.device = config.get("device", "cuda")
        self.action_horizon = config.get("action_horizon", 50)
        img_size = config.get("img_size", 224)

        # ── Lazy import: only triggered in load(), does not pollute global namespace ──
        import torch
        import torchvision.transforms as T
        from fastwam import FastWAMModel

        self.model = FastWAMModel.load_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # FastWAM standard preprocessing pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,).

        Depends on obs fields:
            rgb_static: (H, W, 3) static camera image
            language:   str task description
            proprio:    (7,) current joint position
        """
        # ── Input validation ──
        if obs.rgb_static is None:
            raise ValueError("FastWAMAdapter requires obs.rgb_static")
        if obs.language is None:
            raise ValueError("FastWAMAdapter requires obs.language")
        if obs.proprio is None:
            raise ValueError("FastWAMAdapter requires obs.proprio")

        import torch

        # ── Image preprocessing ──
        rgb = self.transform(obs.rgb_static)          # (3, H, W)
        rgb = rgb.unsqueeze(0).to(self.device)        # (1, 3, H, W)

        # ── Model inference ──
        with torch.no_grad():
            # FastWAM infer returns (1, action_horizon, 7) joint trajectory
            raw_action = self.model.infer(
                image=rgb,
                instruction=[obs.language or ""],
            )

        # ── Convert to numpy ──
        if isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.cpu().numpy()

        # raw_action shape: (1, H, 7) or (H, 7)
        if raw_action.ndim == 3:
            first_step = raw_action[0, 0]  # Take first prediction step
        else:
            first_step = raw_action[0]

        # ── joint position -> delta pose conversion ──
        current_joints = obs.proprio                     # (7,)
        delta_xyz = first_step[:3] - current_joints[:3]  # first 3 joints ~= position
        delta_rot = np.zeros(3, dtype=np.float32)        # FastWAM has no rotation output
        delta_gripper = float(first_step[6]) - float(current_joints[6]) \
            if len(first_step) > 6 else 0.0

        action = np.array(
            [*delta_xyz.tolist(), *delta_rot.tolist(), delta_gripper],
            dtype=np.float64,
        )
        assert action.shape == (STANDARD_ACTION_DIM,), f"Expected (7,), got {action.shape}"
        return action

    def reset(self) -> None:
        """Called at the start of each episode."""
        import torch
        if self.model is not None:
            self.model.reset_history()
