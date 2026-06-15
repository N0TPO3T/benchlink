"""
Canonical observation schema (CanonicalObs).

The data contract between BenchmarkRunner and ModelAdapter.
All fields are either populated or None.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class CanonicalObs:
    """Standard observation dict — intermediate representation between
    BenchmarkRunner and ModelAdapter."""

    # === Vision ===
    rgb_static: Optional[np.ndarray] = None   # (H, W, 3) fixed-view RGB
    rgb_gripper: Optional[np.ndarray] = None  # (H, W, 3) wrist camera RGB
    depth: Optional[np.ndarray] = None         # (H, W)    depth map

    # === Robot state ===
    proprio: Optional[np.ndarray] = None       # (N,) joint positions or end-effector pose

    # === Tactile ===
    tactile_img: Optional[np.ndarray] = None   # (H, W, 3)   tactile image
    tactile_feat: Optional[np.ndarray] = None  # (64,)       pre-extracted tactile features
    tactile_depth: Optional[np.ndarray] = None # (H, W)      tactile depth
    tactile_force: Optional[np.ndarray] = None # (H, W, 3)   tactile force field

    # === Language ===
    language: Optional[str] = None             # task description text
    lang_embed: Optional[np.ndarray] = None    # (512,) pre-computed instruction embedding

    # === Extension fields ===
    extra: Dict[str, Any] = field(default_factory=dict)


# Standard action space
#   action: np.ndarray, shape=(7,)
#   [dx, dy, dz, droll, dpitch, dyaw, gripper]
#   First 6 dims: end-effector delta pose
#   7th dim: gripper in [0, 1]
STANDARD_ACTION_DIM = 7
STANDARD_ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def is_valid_standard_action(action: np.ndarray) -> bool:
    """Validate standard action format."""
    if not isinstance(action, np.ndarray):
        return False
    if action.shape != (STANDARD_ACTION_DIM,):
        return False
    if not np.issubdtype(action.dtype, np.floating):
        return False
    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
        return False
    return True
