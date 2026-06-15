#!/usr/bin/env python3
"""
UniTac_ECF -> TactileAdapter.

UniTac_ECF unified entry point:
  - Selects AnyTouch / Sparsh / T3 encoder via config["model_type"]
  - Provides unified load() / encode() interface
  - Supports registry-based auto-discovery

Usage:
    adapter = UniTac_ECFAdapter()
    adapter.load("", {"model_type": "sparsh", "device": "cuda"})
    feat = adapter.encode(tactile_img)  # -> (768,)
"""

from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import TactileAdapter
from benchlink.registry import get_tactile, list_tactile_models


class UniTac_ECFAdapter(TactileAdapter):
    """UniTac_ECF unified tactile encoder adapter -- delegates to sub-adapter by model_type."""

    # Sub-adapter name -> display name mapping
    SUPPORTED_BACKENDS = ("anytouch", "sparsh", "t3")

    def __init__(self):
        super().__init__()
        self._backend = None       # TactileAdapter sub-instance
        self._backend_name: str = ""
        self.feat_dim: int = 768

    def load(self, checkpoint: str, config: dict) -> None:
        """Load tactile encoder.

        config required fields:
            model_type: Sub-adapter name (e.g. "sparsh", "anytouch", "t3")
        config optional fields:
            device:    Inference device (default "cuda")
            feat_dim:  Feature dimension (default 768)
            (Other parameters are passed through to the sub-adapter)
        """
        model_type = config.get("model_type", "sparsh")
        self.device = config.get("device", "cuda")
        self.feat_dim = config.get("feat_dim", 768)

        if model_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        # Get sub-adapter class via registry and instantiate
        backend_cls = get_tactile(model_type)
        self._backend = backend_cls()
        self._backend_name = model_type

        # Pass through checkpoint and config (unknown fields are automatically ignored by sub-adapter)
        self._backend.load(checkpoint, config)
        self.model = self._backend.model

        self.config = config

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector.

        Delegates to the sub-adapter's encode().
        """
        if self._backend is None:
            raise RuntimeError("UniTac_ECFAdapter not loaded. Call load() first.")
        return self._backend.encode(tactile_img)
