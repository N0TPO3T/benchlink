#!/usr/bin/env python3
"""
Motus persistent inference server.

Runs inside the motus Docker container, communicates via stdin/stdout JSON Lines protocol.
Uses Motus model weights for inference.

Protocol (JSON Lines):
    stdin  ←  {"obs": {"rgb_static": [[H,W,3], ...], "language": str}, "id": N}
    stdout →  {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper], "id": N}
    stdout →  {"status": "ready"}                         # Ready signal
    stdin  ←  {"reset": true, "id": N}                    # Reset
    stdin  ←  {"stop": true, "id": N}                     # Shut down
"""

import json
import sys
import argparse
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


class MotusInferenceServer:
    """Motus inference server, wrapping model loading and inference loop."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.config = None
        self.is_ready = False

        # Language encoding cache  [FIX: key=str, value=Tensor(seq_len, hidden_dim)]
        self._language_cache = {}

        # [FIX] T5 encoder + VLM processor (initialized in _load_model)
        self._t5_encoder = None
        self._vlm_processor = None

        # [FIX] Zero embedding fallback (pre-computed during init)
        self._zero_lang_emb = None

        # Load model
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_model()
        else:
            print("[Motus-Server] No checkpoint found, starting in stub mode.", file=sys.stderr)

    # ────────────────────────────
    # Model loading  [FIX: Rewritten]
    # ────────────────────────────
    def _load_model(self):
        """
        Load Motus model + T5 encoder + VLM processor.
        Three components initialized in sequence, T5 and VLM on CPU to reduce GPU memory usage.
        """
        sys.path.insert(0, "/workspace")
        sys.path.insert(0, "/workspace/baseline/Motus-main")

        # === [FIX] Step 0: Read actual training params from checkpoint ===
        ckpt_config_path = self.checkpoint_path.parent / "config.json"
        ckpt_cfg = {}
        if ckpt_config_path.exists():
            with open(ckpt_config_path) as f:
                ckpt_cfg = json.load(f).get("common", {})
            print(f"[Motus-Server] Loaded checkpoint config: {ckpt_cfg}", file=sys.stderr)

        # === Step 1: Load Motus model ===
        try:
            from models.motus import Motus, MotusConfig

            wan_dir = "/checkpoints/Wan2.2-TI2V-5B"
            vae_path = os.path.join(wan_dir, "Wan2.2_VAE.pth")
            _wan_config_path = wan_dir
            vlm_path = "/checkpoints/Qwen3-VL-2B-Instruct"

            # [FIX] Read params from checkpoint config, use sensible defaults when missing
            nvf = ckpt_cfg.get("num_video_frames", 8)
            vafr = ckpt_cfg.get("video_action_freq_ratio", 2)
            act_dim = ckpt_cfg.get("action_dim", 14)
            state_dim = ckpt_cfg.get("state_dim", 14)
            v_h = ckpt_cfg.get("video_height", 384)
            v_w = ckpt_cfg.get("video_width", 320)

            self.config = MotusConfig(
                wan_checkpoint_path=wan_dir,
                vae_path=vae_path,
                wan_config_path=wan_dir,
                vlm_checkpoint_path=vlm_path,
                load_pretrained_backbones=False,
                video_precision="bfloat16",
                video_height=v_h,
                video_width=v_w,
                num_video_frames=nvf,
                video_action_freq_ratio=vafr,
                action_state_dim=state_dim,
                action_dim=act_dim,
                # [FIX: float() wrap to avoid YAML type issues]
                action_expert_norm_eps=float(1e-6),
                und_expert_norm_eps=float(1e-5),
            )
            print(f"[Motus-Server] Config: frames={nvf}, freq_ratio={vafr}, "
                  f"action_chunk={self.config.action_chunk_size}", file=sys.stderr)

            self.model = Motus(self.config)
            self.model.eval()

            # [FIX: try/except wrap checkpoint loading, allow partial mismatch]
            print(f"[Motus-Server] Loading checkpoint: {self.checkpoint_path}", file=sys.stderr)
            sys.stderr.flush()
            try:
                self.model.load_checkpoint(str(self.checkpoint_path), strict=False)
            except RuntimeError as e:
                print(f"[Motus-Server] Checkpoint partial load (non-fatal): {e}", file=sys.stderr)
            print("[Motus-Server] Checkpoint loaded", file=sys.stderr)

            self.model = self.model.to(self.device, dtype=torch.bfloat16)
            print(f"[Motus-Server] Model ready on {self.device}", file=sys.stderr)

        except Exception as e:
            print(f"[Motus-Server] Model load failed ({e}), using stub.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return  # Don't proceed to load T5/VLM

        # === [FIX] Step 2: Load T5 encoder (CPU) ===
        try:
            from bak.wan.modules.t5 import T5EncoderModel
            print("[Motus-Server] Loading T5EncoderModel on CPU...", file=sys.stderr)
            sys.stderr.flush()
            self._t5_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device="cpu",
                checkpoint_path=os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(wan_dir, "google/umt5-xxl"),
            )
            print("[Motus-Server] T5EncoderModel ready", file=sys.stderr)
        except Exception as e:
            print(f"[Motus-Server] T5 load failed: {e}, text encoding disabled", file=sys.stderr)

        # === [FIX] Step 3: Load VLM processor (CPU) ===
        try:
            from transformers import AutoProcessor
            self._vlm_processor = AutoProcessor.from_pretrained(
                vlm_path, trust_remote_code=True)
            print("[Motus-Server] VLM processor ready", file=sys.stderr)
        except Exception as e:
            print(f"[Motus-Server] VLM processor failed: {e}", file=sys.stderr)

        # === [FIX] Step 4: Pre-compute zero embedding fallback ===
        if self._t5_encoder is not None:
            try:
                dummy = self._t5_encoder([""], device="cpu")  # list[tensor(seq_len, hidden_dim)]
                if isinstance(dummy, list) and len(dummy) > 0:
                    dummy_shape = dummy[0].shape  # (seq_len, hidden_dim)
                    zero = torch.zeros(dummy_shape, dtype=torch.bfloat16, device=self.device)
                    self._zero_lang_emb = [zero]
                    print(f"[Motus-Server] Zero lang emb cached: {dummy_shape}", file=sys.stderr)
            except Exception as e:
                print(f"[Motus-Server] Zero emb init skipped: {e}", file=sys.stderr)

        # [FIX] Language cache: limit max entries to prevent GPU memory leak
        self._language_cache_max = 64

        # Mark as ready
        self.is_ready = True

    # ────────────────────────────
    # Text encoding  [FIX: New method]
    # ────────────────────────────
    def _encode_text(self, text: str) -> list:
        """
        Encode text into language embeddings using T5EncoderModel.

        Returns:
            list[torch.Tensor]: [(seq_len, hidden_dim)], or None (on failure)
        """
        if self._t5_encoder is None:
            return None
        try:
            t5_out = self._t5_encoder([text], device="cpu")
            if isinstance(t5_out, list) and len(t5_out) > 0:
                # [FIX: Cast dtype only, keep cache on CPU to prevent GPU memory leak]
                return [t.to(dtype=torch.bfloat16) for t in t5_out]
            elif isinstance(t5_out, torch.Tensor):
                return [t5_out.to(dtype=torch.bfloat16)]
            return None
        except Exception as e:
            print(f"[Motus-Server] Text encoding failed: {e}", file=sys.stderr)
            return None

    # ────────────────────────────
    # VLM input construction  [FIX: New method]
    # ────────────────────────────
    def _build_vlm_inputs(self, image_np: np.ndarray, text: str) -> Optional[dict]:
        """
        Construct VLM input dict using Qwen3-VL processor.

        Args:
            image_np: (H, W, 3) uint8 RGB image
            text: Language instruction

        Returns:
            dict: {input_ids, attention_mask, pixel_values, ...} or None (on failure)
        """
        if self._vlm_processor is None:
            return None
        try:
            pil_img = Image.fromarray(image_np)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": text or ""},
                    {"type": "image", "image": pil_img},
                ]}
            ]
            formatted = self._vlm_processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False)
            encoded = self._vlm_processor(
                text=[formatted], images=[pil_img], return_tensors="pt")
            # Move to GPU
            vlm_inputs = {}
            for k, v in encoded.items():
                if torch.is_tensor(v):
                    dtype = torch.bfloat16 if k == "pixel_values" else None
                    vlm_inputs[k] = v.to(self.device, dtype=dtype) if dtype else v.to(self.device)
                else:
                    vlm_inputs[k] = v
            return vlm_inputs
        except Exception as e:
            print(f"[Motus-Server] VLM build failed: {e}", file=sys.stderr)
            return None

    # ────────────────────────────
    # Inference  [FIX: Rewrote text encoding and VLM parts]
    # ────────────────────────────
    @torch.no_grad()
    def infer(self, obs: dict) -> list:
        """Single-step inference, returns 7-D action."""
        # ---- Image preprocessing ----
        rgb_frames = obs.get("rgb_static", [])
        if isinstance(rgb_frames, list) and len(rgb_frames) > 0:
            if isinstance(rgb_frames[0][0][0], (int, float)):
                rgb_frames = [rgb_frames]

            # Save a copy of raw image for VLM (uint8, [0,255])
            raw_frame = np.array(rgb_frames[0], dtype=np.uint8)

            # Normalize for model input
            first_frame = raw_frame.astype(np.float32)
            if first_frame.max() > 1.0:
                first_frame = first_frame / 255.0
            first_frame = torch.from_numpy(first_frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        elif self.model is None:
            return [0.0] * 7  # stub mode
        else:
            first_frame = None
            raw_frame = None

        # ---- [FIX] Language instruction → T5 embeddings ----
        language = obs.get("language", "")
        lang_emb = None
        if language:
            if language in self._language_cache:
                lang_emb = self._language_cache[language]
            else:
                encoded = self._encode_text(language)
                if encoded is not None:
                    # [FIX: Limit cache size to prevent GPU memory leak]
                    if len(self._language_cache) >= self._language_cache_max:
                        # LRU eviction: remove oldest cached key
                        oldest = next(iter(self._language_cache))
                        del self._language_cache[oldest]
                    self._language_cache[language] = encoded
                    lang_emb = encoded

        # [FIX] Cache miss or encoding failure → zero embedding fallback
        if lang_emb is None:
            lang_emb = self._zero_lang_emb

        # [FIX] No zero embedding either (no T5) → degrade gracefully
        if lang_emb is None:
            return [0.0] * 7

        # ---- [FIX] VLM inputs ----
        vlm_inputs = None
        if raw_frame is not None:
            vlm_inputs = self._build_vlm_inputs(raw_frame, language)

        # ---- Model inference ----
        if self.model is not None and first_frame is not None:
            try:
                kwargs = dict(
                    first_frame=first_frame,
                    state=None,
                    num_inference_steps=10,
                    language_embeddings=lang_emb,
                )
                # [FIX] Only pass vlm_inputs when successfully constructed
                if vlm_inputs is not None:
                    kwargs["vlm_inputs"] = [vlm_inputs]

                action_tensor, next_state = self.model.inference_step(**kwargs)
                action = action_tensor.detach().cpu().numpy().flatten()
                if action.size >= 7:
                    return action[:7].tolist()
            except Exception as e:
                print(f"[Motus-Server] inference_step failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # Degrade gracefully
        return [0.0] * 7

    def reset(self):
        """Reset model state."""
        if hasattr(self.model, "reset"):
            self.model.reset()


def main():
    parser = argparse.ArgumentParser(description="Motus Persistent Inference Server")
    parser.add_argument("--checkpoint", default="/checkpoints/Motus/mp_rank_00_model_states.pt",
                        help="Motus checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Motus-Server] Starting (device={device})...", file=sys.stderr)
    sys.stderr.flush()

    server = MotusInferenceServer(args.checkpoint, device)

    # Ready signal
    print(json.dumps({"status": "ready"}))
    sys.stdout.flush()

    # JSON Lines loop
    req = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("stop"):
                break
            if req.get("reset"):
                server.reset()
                print(json.dumps({"status": "ok", "id": req.get("id", 0)}))
                sys.stdout.flush()
                continue

            obs = req.get("obs", req)
            action = server.infer(obs)
            resp = {"action": action, "id": req.get("id", 0)}
            print(json.dumps(resp))
            sys.stdout.flush()

        except Exception as e:
            tb = traceback.format_exc()
            req_id = req["id"] if req is not None else 0
            resp = {"error": str(e), "traceback": tb, "id": req_id}
            print(json.dumps(resp))
            sys.stdout.flush()

    print("[Motus-Server] Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
