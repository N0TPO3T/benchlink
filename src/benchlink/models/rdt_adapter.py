#!/usr/bin/env python3
"""
RDT (Robotics Diffusion Transformer) -> ModelAdapter.

Loading: directly import RDT model + multimodal encoder

RDT requires conda environment with:
  - torch, diffusers, transformers
  - huggingface_hub
  - PIL, opencv-python

Model path:
  The checkpoint parameter points to the RDT repo root or pretrained_models/rdt-1b

Usage:
    adapter = RDTAdapter()
    adapter.load("/path/to/RoboticsDiffusionTransformer", {
        "repo_root": "/path/to/RDT",
        "device": "cuda:0",
    })
    action = adapter.act(canonical_obs)  # -> (7,)
"""

import sys
from pathlib import Path

import numpy as np

from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs


class RDTAdapter(ModelAdapter):
    """RDT adapter -- direct import model for diffusion inference."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.img_encoder = None
        self.lang_encoder = None
        self.repo_root: str = ""
        self.action_chunk_size: int = 64
        self._cache = {}

    def load(self, checkpoint: str, config: dict) -> None:
        """Load RDT model and multimodal encoders."""
        import torch

        self.repo_root = config.get("repo_root", checkpoint)
        if not self.repo_root:
            raise ValueError(
                "RDTAdapter requires checkpoint or config['repo_root'] "
                "pointing to RDT repo directory"
            )

        rdt_root = Path(self.repo_root)
        model_path = config.get(
            "model_path", str(rdt_root / "pretrained_models" / "rdt-1b")
        )
        self.device = config.get("device", "cuda")
        self.action_chunk_size = config.get("action_chunk", 64)

        # Add RDT repo to sys.path
        rdt_root_str = str(rdt_root.resolve())
        if rdt_root_str not in sys.path:
            sys.path.insert(0, rdt_root_str)

        # -- Load config --
        import yaml
        cfg_path = config.get("config_path", rdt_root / "configs" / "base.yaml")
        with open(cfg_path) as f:
            self.config = yaml.safe_load(f)
        rdt_cfg = self.config["model"]

        # -- Load RDTRunner --
        from models.rdt_runner import RDTRunner

        print(f"[RDTAdapter] Loading model from {model_path}...")
        sys.stderr.flush()

        self.model = RDTRunner.from_pretrained(
            model_path,
            action_dim=rdt_cfg["state_token_dim"],
            pred_horizon=self.action_chunk_size,
            config=rdt_cfg,
            lang_token_dim=rdt_cfg["lang_token_dim"],
            img_token_dim=rdt_cfg["img_token_dim"],
            state_token_dim=rdt_cfg["state_token_dim"],
            max_lang_cond_len=config.get("max_lang_len", 512),
            img_cond_len=config.get(
                "img_cond_len",
                rdt_cfg.get("img_history_size", 2) * rdt_cfg.get("num_cameras", 1),
            ),
            dtype=torch.bfloat16,
        )
        self.model.to(self.device).eval()

        # -- Load multimodal encoders --
        self._load_encoders(rdt_cfg)

        print(f"[RDTAdapter] Model loaded on {self.device}")
        print(f"[RDTAdapter] Action chunk: {self.action_chunk_size}")
        self.config = config

    def _load_encoders(self, rdt_cfg: dict):
        """Load pretrained vision and language encoders."""

        img_token_dim = rdt_cfg.get("img_token_dim", 1152)
        if img_token_dim == 1152:
            from models.multimodal_encoder.siglip_encoder import SigLipEncoder
            self.img_encoder = SigLipEncoder(
                model_path=str(
                    Path(self.repo_root) / "google" / "siglip-so400m-patch14-384"
                )
            )
        else:
            from models.multimodal_encoder.clip_encoder import CLIPEncoder
            self.img_encoder = CLIPEncoder()

        from models.multimodal_encoder.t5_encoder import T5Encoder
        self.lang_encoder = T5Encoder(
            model_path=rdt_cfg.get("lang_encoder_path", "google/flan-t5-xl")
        )

        self.img_encoder.to(self.device)
        self.lang_encoder.to(self.device)

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,)."""
        import torch

        if self.model is None:
            raise RuntimeError("RDTAdapter model not loaded. Call load() first.")

        if obs.rgb_static is None and obs.rgb_gripper is None:
            raise ValueError("RDTAdapter requires at least one camera image")

        with torch.no_grad():
            # 1. Image encoding
            img_tensor = self._preprocess_image(obs)
            img_tokens = self.img_encoder(img_tensor.to(self.device))

            # 2. Language encoding
            lang_text = obs.language or ""
            lang_tokens, lang_attn_mask = self.lang_encoder([lang_text])

            # 3. State/action encoding
            state = obs.proprio if obs.proprio is not None else np.zeros(8)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
            state_tokens = self.model.state_adaptor(state_tensor.to(self.device))

            # 4. Action mask
            state_dim = self.config["model"]["state_token_dim"]
            action_mask = torch.ones(
                1, 1, state_dim, dtype=torch.float32, device=self.device
            )

            # 5. Diffusion inference
            ctrl_freq = torch.full((1,), 50, dtype=torch.float32, device=self.device)

            action_chunk = self.model.predict_action(
                lang_tokens=lang_tokens.to(self.device),
                lang_attn_mask=lang_attn_mask.to(self.device),
                img_tokens=img_tokens.to(self.device),
                state_tokens=state_tokens.to(self.device),
                action_mask=action_mask,
                ctrl_freqs=ctrl_freq,
            )

        # 6. Take first frame -> (7,)
        action = action_chunk[0, 0].cpu().numpy()
        return action[:7].astype(np.float64)

    def _preprocess_image(self, obs: CanonicalObs):
        """Preprocess image into model input format."""
        import torchvision.transforms as T

        img = obs.rgb_static if obs.rgb_static is not None else obs.rgb_gripper

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(img).unsqueeze(0).unsqueeze(0)

    def reset(self) -> None:
        """Clear cached state."""
        self._cache.clear()

    def close(self) -> None:
        """Clean up resources."""
        self._cache.clear()
        self.model = None
        self.img_encoder = None
        self.lang_encoder = None
