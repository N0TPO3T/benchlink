#!/usr/bin/env python3
"""
Sparsh (DINOv2) -> TactileAdapter.

Sparsh (CoRL 2024, Meta):
  - DINOv2-based universal tactile representation model
  - Self-supervised training (iBOT + DINO loss), pretrained on 2M+ tactile images
  - Input: tactile image (H, W, 3)
  - Output: feature vector (D,) — default 768 (ViT-B/14) or 1024 (ViT-L/14)

This adapter supports two loading modes:
  1. Sparsh-specific checkpoint (contains encoder weights)
  2. Pure timm DINOv2 backbone (no Sparsh finetuning, fallback)

Notes:
  - The official Sparsh library (tactile_ssl) is a training framework, not an inference API
  - This adapter directly uses timm's DINOv2 + optional Sparsh checkpoint weight loading
  - The tactile_ssl model interface can be enabled via sparsh_repo config

Usage:
    adapter = SparshAdapter()
    adapter.load("/path/to/sparsh_checkpoint.pth", {"device": "cuda"})
    feat = adapter.encode(tactile_img)  # -> (768,)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import TactileAdapter


class SparshAdapter(TactileAdapter):
    """Sparsh tactile encoder adapter -- based on DINOv2 backbone."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.feat_dim: int = 768
        self.backbone_name: str = "vit_base"
        self.use_sparsh_repo: bool = False

    def load(self, checkpoint: str, config: dict) -> None:
        """Load Sparsh/DINOv2 encoder weights.

        Args:
            checkpoint: Path to Sparsh checkpoint (.pth), empty string for timm DINOv2
            config: Configuration dict with supported fields:
                device:         Inference device (default "cuda")
                backbone:       ViT scale (default "vit_base", options "vit_large", "vit_giant")
                feat_dim:       Output feature dimension (default 768)
                img_size:       Input image size (default 224)
                sparsh_repo:    tactile_ssl repo path (for loading Sparsh custom model)
        """
        self.device = config.get("device", "cuda")
        self.backbone_name = config.get("backbone", "vit_base")
        self.feat_dim = config.get("feat_dim", 768)
        img_size = config.get("img_size", 224)

        import timm
        import torch
        import torchvision.transforms as T

        # -- Backbone name mapping --
        dino_model_map = {
            "vit_base":   "vit_base_patch14_reg4_dinov2.lvd142m",
            "vit_large":  "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_giant":  "vit_giant_patch14_reg4_dinov2.lvd142m",
        }
        timm_model_name = dino_model_map.get(self.backbone_name, "vit_base_patch14_reg4_dinov2.lvd142m")

        # -- Attempt to load Sparsh-specific checkpoint --
        checkpoint_path = str(checkpoint) if checkpoint else ""
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.model = timm.create_model(
                    timm_model_name, pretrained=False, num_classes=0,
                )
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if "encoder" in state_dict:
                    state_dict = state_dict["encoder"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                model_keys = set(self.model.state_dict().keys())
                load_keys = {k: v for k, v in state_dict.items()
                             if k in model_keys}
                if load_keys:
                    self.model.load_state_dict(load_keys, strict=False)
                    print(f"[SparshAdapter] Loaded Sparsh checkpoint: {checkpoint_path}")
                else:
                    print(f"[SparshAdapter] Checkpoint key mismatch, using timm pretrained")
                    self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
            except Exception as e:
                print(f"[SparshAdapter] Checkpoint load failed ({e}), using timm pretrained")
                self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        else:
            # Pure timm DINOv2
            self.model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
            print(f"[SparshAdapter] Using timm DINOv2 backbone: {timm_model_name}")

        self.model.to(self.device)
        self.model.eval()

        # -- DINOv2 standard preprocessing (uses model default input size) --
        # timm stores img_size in patch_embed
        model_img_size = None
        if hasattr(self.model, "patch_embed"):
            model_img_size = getattr(self.model.patch_embed, "img_size", None)
        if model_img_size is None:
            model_img_size = [img_size, img_size]
        resize_size = model_img_size[0] if isinstance(model_img_size, (list, tuple)) else img_size

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print(f"[SparshAdapter] Using image size: {resize_size}x{resize_size}")

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector.

        Args:
            tactile_img: (H, W, 3) uint8 tactile image

        Returns:
            ndarray shape=(feat_dim,)
        """
        import torch

        # -- Preprocessing --
        img = self.transform(tactile_img)              # (3, img_size, img_size)
        img = img.unsqueeze(0).to(self.device)          # (1, 3, img_size, img_size)

        # -- DINOv2 encoding -> [CLS] token --
        with torch.no_grad():
            # forward_features returns an intermediate dict
            output = self.model.forward_features(img)
            # DINOv2 timm returns: {"x_norm_patchtokens": ..., "x_norm_regtokens": ..., "x_norm_clstoken": ...}
            if isinstance(output, dict):
                for key in ["x_norm_clstoken", "x_norm_patchtokens", "x_norm_regtokens"]:
                    if key in output:
                        feat = output[key]
                        break
                else:
                    feat = list(output.values())[0]
            else:
                feat = output

        if feat.ndim == 2:
            feat = feat[0] if feat.shape[0] == 1 else feat[:, 0]
        elif feat.ndim == 3:
            feat = feat[0]
        feat = feat.flatten()[:self.feat_dim].cpu().numpy()  # (feat_dim,)

        return feat
