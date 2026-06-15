#!/usr/bin/env python3
"""
VLA-Touch -> ModelAdapter + TactileAdapter.

VLA-Touch (Octopi-based, Rutgers):
  - Qwen2VL (7B/8B) vision-language backbone + Sparsh tactile encoder
  - Input: RGB image + language instruction + tactile image
  - Output: 7-D standard action [dx, dy, dz, droll, dpitch, dyaw, gripper]

Architecture:
  Qwen2VL (VLM backbone):
    - Image Encoder: processes RGB image -> visual embeddings
    - LLM: processes text + visual embeddings -> joint representation

  Sparsh (tactile encoder):
    - DINOv2 backbone + optional Sparsh checkpoint
    - Tactile image -> [CLS] feature

  Tactile Projection:
    - nn.Linear(Sparsh_feat_dim, Qwen_hidden_dim)
    - Aligns tactile features to Qwen embedding space

  Action Head:
    - Mode 1 (regression, default): Qwen last hidden -> MLP -> 7-D continuous action
    - Mode 2 (token_decode): Qwen.generate() -> action tokens -> 7-D continuous action

  Data flow (act, regression mode):
    rgb_static --|-> QwenVL Image Encoder --|
    language   --|-> QwenVL Tokenizer   ---|
    tactile_img-|-> Sparsh Encoder -|-> Proj -|
                                              |
                                         Qwen LLM
                                              |
                               last_hidden + tactile_feat -> concat
                                              |
                                      Action Head MLP
                                              |
                                      ndarray (7,)

  Data flow (encode):
    tactile_img -> Sparsh Encoder -> CLS token -> ndarray (feat_dim,)

Usage:
    adapter = VLA_TouchAdapter()
    adapter.load("/path/to/vla_touch_checkpoint.pth", {"device": "cuda"})
    action = adapter.act(canonical_obs)  # -> (7,)
    feat   = adapter.encode(tactile_img) # -> (768,)

Dependencies:
    pip install transformers timm torchvision
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import ModelAdapter, TactileAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class VLA_TouchAdapter(ModelAdapter, TactileAdapter):
    """VLA-Touch adapter -- dual interface: ModelAdapter (act/reset) + TactileAdapter (encode)."""

    def __init__(self):
        # Dual inheritance: ModelAdapter.__init__ -> TactileAdapter.__init__
        # Both set self.device = "cpu", self.model = None
        super().__init__()
        self.qwen = None                # Qwen2VLForConditionalGeneration
        self.qwen_processor = None      # Qwen2VLProcessor
        self.tactile_encoder = None     # timm DINOv2 model (Sparsh)
        self.tactile_proj = None        # nn.Linear(feat_dim, qwen_hidden)
        self.action_head = None         # MLP action head (regression mode)
        self.transform_tactile = None   # Tactile image preprocessing
        self.transform_rgb = None       # RGB image preprocessing (Qwen-compatible)

        # Tactile encoder config (consistent with SparshAdapter)
        self.feat_dim: int = 768
        self.backbone_name: str = "vit_base"

        # Action decoding config
        self.action_decode_mode: str = "regression"   # "regression" | "token_decode"
        self.max_action_tokens: int = 7
        self._action_low: np.ndarray = np.array(
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0], dtype=np.float64
        )
        self._action_high: np.ndarray = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float64
        )

        # Repo paths
        self._repo_root: Optional[Path] = None

    def load(self, checkpoint: str, config: dict) -> None:
        """Load VLA-Touch model weights.

        Args:
            checkpoint: Path to VLA-Touch checkpoint (.pth).
                        Contains qwen LoRA / tactile_encoder / tactile_proj / action_head.
                        Empty string loads only pretrained Qwen2VL + timm DINOv2.
            config: Configuration dict with supported fields:
                device:               Inference device (default "cuda")
                qwen_model_name:      Qwen2VL model name (default "Qwen/Qwen2-VL-7B-Instruct")
                backbone:            Sparsh backbone (default "vit_base")
                feat_dim:            Tactile feature dimension (default 768)
                action_decode_mode:  Action decoding mode (default "regression")
                action_low:          Action lower bound 7-D list (default [-0.5,...])
                action_high:         Action upper bound 7-D list (default [0.5,...])
                num_action_bins:     Number of bins per dimension in token_decode mode (default 256)
                max_action_tokens:   Number of action tokens (default 7)
                img_size:            Input image size (default 224)
                vla_repo:            VLA-Touch repo path (optional)
        """
        self.device = config.get("device", "cuda")
        qwen_name = config.get("qwen_model_name", "Qwen/Qwen2-VL-7B-Instruct")
        self.backbone_name = config.get("backbone", "vit_base")
        self.feat_dim = config.get("feat_dim", 768)
        self.action_decode_mode = config.get("action_decode_mode", "regression")
        self.max_action_tokens = config.get("max_action_tokens", 7)

        # Action bounds
        if "action_low" in config:
            self._action_low = np.asarray(config["action_low"], dtype=np.float64)
        if "action_high" in config:
            self._action_high = np.asarray(config["action_high"], dtype=np.float64)

        # -- Add repo to sys.path --
        repo_root = config.get("vla_repo", "")
        if repo_root:
            self._repo_root = Path(repo_root)
            repo_str = str(self._repo_root)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        import torch
        import torch.nn as nn
        import timm
        from transformers import (
            Qwen2VLForConditionalGeneration,
            Qwen2VLProcessor,
        )

        # ================================================
        # 1. Load Qwen2VL
        # ================================================
        attn_impl = config.get("attn_implementation", "sdpa")
        print(f"[VLA_TouchAdapter] Loading Qwen2VL: {qwen_name} "
              f"(attn={attn_impl}, device={self.device})")
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self.qwen_processor = Qwen2VLProcessor.from_pretrained(qwen_name)
        qwen_hidden = self.qwen.config.hidden_size  # 4096 for 7B

        # -- Qwen image preprocessing --
        # Qwen2VLProcessor manages image_processor internally, no extra transform needed
        self.transform_rgb = None  # Using processor built-in

        # ================================================
        # 2. Load tactile encoder (Sparsh DINOv2)
        # ================================================
        img_size = config.get("img_size", 224)
        dino_model_map = {
            "vit_base":   "vit_base_patch14_reg4_dinov2.lvd142m",
            "vit_large":  "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_giant":  "vit_giant_patch14_reg4_dinov2.lvd142m",
        }
        timm_model = dino_model_map.get(
            self.backbone_name,
            "vit_base_patch14_reg4_dinov2.lvd142m",
        )

        self.tactile_encoder = timm.create_model(
            timm_model, pretrained=True, num_classes=0,
        )
        self.tactile_encoder.to(self.device)
        self.tactile_encoder.eval()

        # -- Tactile projection layer (Sparsh feat_dim -> Qwen hidden_size) --
        self.tactile_proj = nn.Linear(self.feat_dim, qwen_hidden)
        self.tactile_proj.to(self.device)

        # -- Tactile image preprocessing (consistent with SparshAdapter) --
        import torchvision.transforms as T
        self.transform_tactile = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # ================================================
        # 3. Load Action Head (regression mode)
        # ================================================
        if self.action_decode_mode == "regression":
            # action_head input: Qwen hidden + Projected Tactile (qwen_hidden)
            self.action_head = nn.Sequential(
                nn.Linear(qwen_hidden + qwen_hidden, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, STANDARD_ACTION_DIM),
            )
            self.action_head.to(self.device)
        elif self.action_decode_mode == "token_decode":
            raise NotImplementedError(
                "token_decode mode is not yet implemented. "
                "Use 'regression' mode instead."
            )
        else:
            raise ValueError(
                f"Unknown action_decode_mode: {self.action_decode_mode}. "
                f"Expected 'regression' or 'token_decode'"
            )

        # ================================================
        # 4. Load checkpoint weights
        # ================================================
        ckpt_path = str(checkpoint) if checkpoint and Path(checkpoint).exists() else ""
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
        elif checkpoint:
            print(f"[VLA_TouchAdapter] Warning: checkpoint not found: {checkpoint}, "
                  "using pretrained weights only")

        self.qwen.eval()

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference, returns standard action (7,).

        Depends on obs fields:
            rgb_static:   (H, W, 3) fixed-view image
            language:     task description text
            tactile_img:  (H, W, 3) tactile image (optional)
        """
        import torch

        if obs.rgb_static is None:
            raise ValueError("VLA_TouchAdapter requires obs.rgb_static")

        # ================================================
        # 1. Qwen vision-language inference
        # ================================================
        prompt = f"Task: {obs.language or ''}\nWhat action should the robot take?\nAction:"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": obs.rgb_static},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.qwen_processor(
            text=[text],
            images=[obs.rgb_static],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # ================================================
        # 2. Tactile encoding
        # ================================================
        # Supports both tactile_img (raw image) and tactile_feat (pre-extracted features)
        if obs.tactile_img is not None:
            tactile_feat = self._encode_tactile_to_tensor(obs.tactile_img)
        elif obs.tactile_feat is not None:
            tactile_feat = torch.as_tensor(obs.tactile_feat, dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
        else:
            tactile_feat = torch.zeros(1, self.feat_dim, device=self.device)

        # ================================================
        # 3. Action decoding
        # ================================================
        if self.action_decode_mode == "regression":
            action = self._act_regression(inputs, tactile_feat)
        else:
            action = self._act_token_decode(inputs, tactile_feat)

        return action

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """Tactile image -> feature vector.

        Args:
            tactile_img: (H, W, 3) uint8 tactile image

        Returns:
            ndarray shape=(feat_dim,), e.g. (768,)
        """
        feat = self._encode_tactile_to_tensor(tactile_img)
        return feat.squeeze(0).cpu().numpy()

    def reset(self) -> None:
        """Reset internal state (currently no cached state)."""
        pass

    # -- Internal methods --

    def _load_checkpoint(self, ckpt_path: str) -> None:
        """Load VLA-Touch checkpoint weights."""
        import torch

        state_dict = torch.load(ckpt_path, map_location="cpu")
        loaded_any = False

        # -- Tactile encoder --
        enc_key = state_dict.get("tactile_encoder",
                                  state_dict.get("sparsh", None))
        if enc_key is not None:
            tact_keys = {k: v for k, v in enc_key.items()
                         if k in self.tactile_encoder.state_dict()}
            if tact_keys:
                self.tactile_encoder.load_state_dict(tact_keys, strict=False)
                loaded_any = True
                print(f"[VLA_TouchAdapter] Loaded tactile encoder ({len(tact_keys)} keys)")

        # -- Tactile projection layer --
        proj_key = state_dict.get("tactile_projection",
                                   state_dict.get("tactile_proj", None))
        if proj_key is not None:
            self.tactile_proj.load_state_dict(
                {k: v for k, v in proj_key.items()
                 if k in self.tactile_proj.state_dict()},
                strict=False,
            )
            loaded_any = True
            print("[VLA_TouchAdapter] Loaded tactile projection")

        # -- Action Head --
        head_key = state_dict.get("action_head", None)
        if head_key is not None and self.action_head is not None:
            self.action_head.load_state_dict(
                {k: v for k, v in head_key.items()
                 if k in self.action_head.state_dict()},
                strict=False,
            )
            loaded_any = True
            print("[VLA_TouchAdapter] Loaded action head")

        if not loaded_any:
            print("[VLA_TouchAdapter] Warning: No matching keys found in checkpoint, "
                  "using pretrained weights only")

    def _act_regression(
        self,
        inputs: dict,
        tactile_feat,
    ) -> np.ndarray:
        """regression mode: Qwen hidden + tactile(projected) -> MLP -> 7-D action.

        Uses Qwen2VL as a feature extractor, takes the last-layer hidden state,
        projects tactile features to the Qwen embedding space via tactile_proj,
        concatenates them, and regresses the continuous action through action_head.
        """
        import torch

        with torch.no_grad():
            outputs = self.qwen(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            # -- Guard check: hidden_states may be None --
            if outputs.hidden_states is None:
                raise RuntimeError(
                    "Qwen2VL did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported."
                )

            # (1, seq_len, hidden_size) -> take last token
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # (1, qwen_hidden)

            # -- Project tactile features to Qwen embedding space --
            # tactile_feat: (1, feat_dim) -> (1, qwen_hidden)
            tactile_projected = self.tactile_proj(tactile_feat)

            # Concatenate Qwen hidden + projected tactile
            combined = torch.cat([last_hidden, tactile_projected], dim=-1)

            # Action head inference
            raw_action = self.action_head(combined)  # (1, 7)

        action = raw_action.squeeze(0).cpu().numpy()

        # Clipping + boundary protection
        action = np.clip(action, self._action_low, self._action_high)
        return action.astype(np.float64)

    def _act_token_decode(
        self,
        inputs: dict,
        tactile_feat,
    ) -> np.ndarray:
        """token_decode mode: interface stub, not yet implemented."""
        raise NotImplementedError(
            "token_decode mode is not yet implemented. "
            "Use action_decode_mode='regression' instead."
        )

    def _encode_tactile_to_tensor(self, tactile_img: np.ndarray):
        """Tactile image -> (1, feat_dim) tensor (preserves batch dimension)."""
        import torch

        img = self.transform_tactile(tactile_img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.tactile_encoder.forward_features(img)
            if isinstance(output, dict):
                # Try CLS token, if absent use patchtokens and pool
                feat = output.get("x_norm_clstoken", None)
                if feat is None:
                    feat = output.get("x_norm_patchtokens", None)
                    if feat is None:
                        raise KeyError(
                            f"DINOv2 forward_features dict has unexpected keys: "
                            f"{list(output.keys())}"
                        )
            else:
                feat = output

        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat)

        # Preserve batch dimension: (1, feat_dim)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        return feat  # (1, feat_dim)

    # -- Utility methods --

    def set_action_bounds(self, low: list, high: list) -> None:
        """Set action clipping bounds."""
        self._action_low = np.asarray(low, dtype=np.float64)
        self._action_high = np.asarray(high, dtype=np.float64)
