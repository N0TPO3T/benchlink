#!/usr/bin/env python3
"""
AnyTouch -> TactileAdapter.

AnyTouch (ICLR 2025):
  - Tactile representation model, two-stage training: Stage 1 MAE pretrain + Stage 2 CLIP alignment
  - Input: tactile image (H, W, 3)
  - Output: feature vector (D,) -- default 768 (CLIP ViT-L/14 backbone)

Actual repo path: UniTac_ECF/anytouch/
  Stage 1: model/mae_model.py -> TactileMAE (MAE pretraining)
  Stage 2: model/process_clip.py -> CLIP vision encoder

Usage:
    adapter = AnyTouchAdapter()
    adapter.load("/path/to/anytouch_stage2.pth", {"device": "cuda"})
    feat = adapter.encode(tactile_img)  # -> (768,)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import TactileAdapter


class AnyTouchAdapter(TactileAdapter):
    """AnyTouch tactile encoder adapter."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.feat_dim: int = 768
        self._repo_root: Optional[Path] = None

    def load(self, checkpoint: str, config: dict) -> None:
        """Load AnyTouch encoder weights.

        Args:
            checkpoint: Path to AnyTouch checkpoint (.pth).
                        Supports Stage 1 (MAE) or Stage 2 (CLIP) format.
                        Empty string for random initialization (testing only).
            config: Config dict, supported fields:
                device:        Inference device (default "cuda")
                img_size:      Input image size (default 224)
                feat_dim:      Output feature dimension (default 768)
                anytouch_repo: Path to anytouch repo (default auto-discovered from sys.path)
                stage:         Model stage (default 2, optional 1)
        """
        self.device = config.get("device", "cuda")
        self.feat_dim = config.get("feat_dim", 768)
        img_size = config.get("img_size", 224)
        stage = config.get("stage", 2)

        # ── Add anytouch repo to sys.path ──
        repo_root = config.get("anytouch_repo", "")
        if repo_root:
            self._repo_root = Path(repo_root)
            repo_str = str(self._repo_root / "anytouch")
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        import torchvision.transforms as T

        if stage == 2:
            self._load_stage2(checkpoint, img_size)
        else:
            self._load_stage1(checkpoint, img_size)

        self.model.to(self.device)
        self.model.eval()

        # ── CLIP standard preprocessing ──
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector.

        Args:
            tactile_img: (H, W, 3) uint8 tactile image

        Returns:
            ndarray shape=(feat_dim,), e.g. (768,)
        """
        import torch

        img = self.transform(tactile_img)              # (3, img_size, img_size)
        img = img.unsqueeze(0).to(self.device)          # (1, 3, img_size, img_size)

        with torch.no_grad():
            feat = self.model(img)                     # (1, D)

        if isinstance(feat, torch.Tensor):
            feat = feat.squeeze(0).cpu().numpy()       # (D,)

        return feat

    # ── Load implementations ──

    def _load_stage2(self, checkpoint: str, img_size: int) -> None:
        """Load Stage 2 CLIP alignment model."""
        import torch
        import torch.nn as nn
        from transformers import CLIPVisionModel
        from transformers.models.clip.configuration_clip import CLIPConfig

        # CLIP visual encoder config
        clip_cfg = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14-336")
        clip_cfg.vision_config.image_size = img_size

        # AnyTouch Stage 2 uses CLIP visual encoder + tactile projection
        vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            config=clip_cfg.vision_config,
        )

        hidden_size = clip_cfg.vision_config.hidden_size  # 1024 for ViT-L

        class CLIPVisionWrapper(nn.Module):
            """Wrapper for CLIPVisionTransformer that extracts pooler_output and projects."""
            def __init__(self, vision_model, out_dim):
                super().__init__()
                self.vision_model = vision_model
                self.ln = nn.LayerNorm(hidden_size)
                self.proj = nn.Linear(hidden_size, out_dim)

            def forward(self, x):
                outputs = self.vision_model(x, output_hidden_states=False)
                # outputs.pooler_output: (B, hidden_size) contains CLS token + projection
                # Alternatively, use last_hidden_state[:, 0] for CLS token
                cls_token = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
                return self.proj(self.ln(cls_token))

        self.model = CLIPVisionWrapper(vision_model.vision_model, self.feat_dim)

        # Load checkpoint
        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Stage 2 checkpoint format: may have vision_model prefix
            if any(k.startswith("vision_model") for k in state_dict):
                # Strip "vision_model." prefix to match CLIPVisionWrapper's state_dict
                clean = {k.replace("vision_model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean, strict=False)
            elif any(k.startswith("model.") for k in state_dict):
                clean = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
            print(f"[AnyTouchAdapter] Loaded Stage 2 checkpoint: {checkpoint}")
        else:
            print("[AnyTouchAdapter] No checkpoint, using pretrained CLIPVisionModel")

    def _load_stage1(self, checkpoint: str, img_size: int) -> None:
        """Load Stage 1 MAE pretrained model (encoder only)."""
        import torch
        from model.mae_model import TactileMAE

        # TactileMAE defaults to ViT-L
        self.model = TactileMAE(
            img_size=img_size,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
        )

        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Stage 1 checkpoint: extract only encoder weights
            encoder_keys = {k: v for k, v in state_dict.items()
                            if k.startswith("encoder.")}
            if encoder_keys:
                self.model.load_state_dict(encoder_keys, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
            print(f"[AnyTouchAdapter] Loaded Stage 1 checkpoint: {checkpoint}")
