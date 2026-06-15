#!/usr/bin/env python3
"""
T3 (Transferable Tactile Transformer) -> TactileAdapter.

T3 (CVPR 2025, MIT CSAIL):
  - Generalizable tactile representation learning framework
  - Architecture: ResNet/CNN/ViT/MAE-ViT backbone + learnable frequency-domain filter + cross-modal alignment MLP
  - Input: tactile image (H, W, 3)
  - Output: feature vector (D,) — default 1024 dims

Supported encoder types:
  - "resnet":   ResNet18/50 + custom head (output 1024)
  - "cnn":      Custom CNN (output 1024)
  - "vit":      ViT-B/16 (output 768)
  - "mae_vit":  MAE-pretrained ViT (output 1024) — recommended

Usage:
    adapter = T3Adapter()
    adapter.load("/path/to/t3_encoder.pth", {"encoder_type": "mae_vit"})
    feat = adapter.encode(tactile_img)  # -> (1024,)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import TactileAdapter


class T3Adapter(TactileAdapter):
    """T3 tactile encoder adapter."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.feat_dim: int = 1024
        self.encoder_type: str = "mae_vit"

    def load(self, checkpoint: str, config: dict) -> None:
        """Load T3 encoder weights.

        Args:
            checkpoint: Path to T3 encoder checkpoint (.pth)
            config: Configuration dict with supported fields:
                device:        Inference device (default "cuda")
                encoder_type:  Encoder type (default "mae_vit")
                              "resnet", "cnn", "vit", "mae_vit"
                feat_dim:      Output feature dimension (default 1024)
                img_size:      Input image size (default 224)
                t3_repo:       T3 repo path (default looked up from sys.path)
        """
        self.device = config.get("device", "cuda")
        self.encoder_type = config.get("encoder_type", "mae_vit")
        self.feat_dim = config.get("feat_dim", 1024)
        img_size = config.get("img_size", 224)

        # -- Add T3 repo to sys.path (via .pth or manually) --
        t3_repo = config.get("t3_repo", "")
        if t3_repo and str(t3_repo) not in sys.path:
            sys.path.insert(0, str(t3_repo))

        import torch
        import torchvision.transforms as T

        # -- Build encoder --
        if self.encoder_type == "mae_vit":
            self.model = self._build_mae_vit_encoder(img_size, checkpoint)
        elif self.encoder_type == "vit":
            self.model = self._build_vit_encoder(img_size, checkpoint)
        elif self.encoder_type == "resnet":
            self.model = self._build_resnet_encoder(checkpoint)
        elif self.encoder_type == "cnn":
            self.model = self._build_cnn_encoder(img_size, checkpoint)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.model.to(self.device)
        self.model.eval()

        # -- Preprocessing --
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # -- encode --

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector.

        Args:
            tactile_img: (H, W, 3) uint8 tactile image

        Returns:
            ndarray shape=(feat_dim,)
        """
        import torch

        img = self.transform(tactile_img)              # (3, img_size, img_size)
        img = img.unsqueeze(0).to(self.device)          # (1, 3, img_size, img_size)

        with torch.no_grad():
            raw_output = self.model(img)                # (1, D) or (D,) or (tuple)

        # -- Normalize output format --
        if isinstance(raw_output, torch.Tensor):
            feat = raw_output
        elif isinstance(raw_output, (tuple, list)):
            # Some T3 encoders return (features, aux_loss) tuple
            feat = raw_output[0]
        elif isinstance(raw_output, dict):
            feat = raw_output.get("features", list(raw_output.values())[0])
        else:
            raise TypeError(f"Unexpected model output type: {type(raw_output)}")

        feat = feat.squeeze(0).cpu().numpy()             # (D,)

        return feat

    # -- Encoder factory methods --

    def _build_mae_vit_encoder(self, img_size: int, checkpoint: str) -> 'torch.nn.Module':
        """Build MAE-pretrained ViT encoder."""
        from t3.models.encoder import MAEViTEncoder

        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        encoder = MAEViTEncoder(
            output_dim=self.feat_dim,
            img_size=img_size,
        )
        if ckpt_path:
            try:
                encoder.load(ckpt_path)
            except Exception as e:
                print(f"[T3Adapter] Checkpoint load failed ({e}), using random init")
        return encoder

    def _build_vit_encoder(self, img_size: int, checkpoint: str) -> 'torch.nn.Module':
        """Build standard ViT encoder."""
        from t3.models.encoder import ViTEncoder

        encoder = ViTEncoder(
            output_dim=self.feat_dim,
            img_size=img_size,
            patch_size=16,
        )
        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            try:
                encoder.load(ckpt_path)
            except Exception as e:
                print(f"[T3Adapter] Checkpoint load failed ({e}), using random init")
        return encoder

    def _build_resnet_encoder(self, checkpoint: str) -> 'torch.nn.Module':
        """Build ResNet encoder."""
        from t3.models.encoder import ResNetEncoder

        encoder = ResNetEncoder(
            output_dim=self.feat_dim,
            model="resnet50",
            pretrained=True,
        )
        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            try:
                encoder.load(ckpt_path)
            except Exception as e:
                print(f"[T3Adapter] Checkpoint load failed ({e}), using ImageNet pretrained")
        return encoder

    def _build_cnn_encoder(self, img_size: int, checkpoint: str) -> 'torch.nn.Module':
        """Build custom CNN encoder."""
        from t3.models.encoder import CNNEncoder

        encoder = CNNEncoder(
            output_dim=self.feat_dim,
            input_channels=3,
            img_size=img_size,
            filters=[64, 128, 256, 512],
        )
        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            try:
                encoder.load(ckpt_path)
            except Exception as e:
                print(f"[T3Adapter] Checkpoint load failed ({e}), using random init")
        return encoder
