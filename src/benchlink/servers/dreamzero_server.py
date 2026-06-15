#!/usr/bin/env python3
"""
DreamZero persistent inference server.

Runs inside the DreamZero Docker container,
communicates via stdin/stdout JSON Lines protocol.

Uses the groot.vla.model.dreamzero.base_vla.VLA model for inference.

Protocol (JSON Lines):
    stdin  ←  {"obs": {"rgb_static": list, "language": str}, "id": N}
    stdout →  {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper], "id": N}
    stdout →  {"status": "ready"}
    stdin  ←  {"reset": true, "id": N}     # Clear latent history
    stdin  ←  {"stop": true, "id": N}      # Shut down server
"""

import json
import sys
import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import torch


DREAMZERO_ROOT = "/workspace/DreamZero"
CHECKPOINT_DIR = "/checkpoints/DreamZero-DROID"


class DreamZeroInferenceServer:
    """DreamZero inference server, wrapping VLA model loading and inference."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.is_ready = False

        # latent history
        self.latent_buffer = []
        self.max_buffer = 4

        # Load model
        self._load_model()

    def _load_model(self):
        """Load DreamZero VLA model."""
        sys.path.insert(0, DREAMZERO_ROOT)

        cp = self.checkpoint_path or CHECKPOINT_DIR

        if not Path(cp).exists():
            print(f"[DreamZero-Server] WARNING: checkpoint not found at {cp}", file=sys.stderr)
            print("[DreamZero-Server] Starting in stub mode (returning zero actions)", file=sys.stderr)
            return

        try:
            from groot.vla.model.dreamzero.base_vla import VLA

            print(f"[DreamZero-Server] Loading VLA model from {cp}...", file=sys.stderr)
            sys.stderr.flush()

            # VLA.from_pretrained only accepts path and config arguments
            # Manually transfer device after loading
            self.model = VLA.from_pretrained(str(cp))
            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_ready = True
            print(f"[DreamZero-Server] VLA model loaded successfully on {self.device}", file=sys.stderr)

            # Print model info
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"[DreamZero-Server] Model size: {param_count:.2f}B params", file=sys.stderr)

        except Exception as e:
            print(f"[DreamZero-Server] Failed to load VLA model: {e}", file=sys.stderr)
            print(f"[DreamZero-Server] Traceback: {traceback.format_exc()}", file=sys.stderr)
            print("[DreamZero-Server] Starting in stub mode.", file=sys.stderr)

    @torch.no_grad()
    def infer(self, obs: dict) -> list:
        """Single-step inference, returns 7-D action."""
        if self.model is None or not self.is_ready:
            return [0.0] * 7

        try:
            # Preprocess input
            inputs = self._preprocess(obs)

            # Use VLA.get_action for inference
            output = self.model.get_action(inputs)
            # BatchFeature is dict-like, extract action
            if isinstance(output, dict):
                # Try common keys
                for key in ["action", "actions", "action_pred", "pred_action"]:
                    if key in output:
                        action = output[key]
                        break
                else:
                    # Take the last tensor value
                    action = list(output.values())[-1]
            elif isinstance(output, torch.Tensor):
                action = output
            else:
                action = output

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()

            action = np.asarray(action, dtype=np.float64).flatten()
            if action.size < 7:
                action = np.pad(action, (0, 7 - action.size))[:7]
            elif action.size > 7:
                action = action[:7]

            # Update latent buffer
            if hasattr(self.model, "get_latent"):
                try:
                    latent = self.model.get_latent()
                    if isinstance(latent, torch.Tensor):
                        self.latent_buffer.append(latent.detach().cpu())
                        if len(self.latent_buffer) > self.max_buffer * 2:
                            self.latent_buffer = self.latent_buffer[-self.max_buffer:]
                except Exception:
                    pass

            return action.tolist()

        except Exception as e:
            print(f"[DreamZero-Server] inference error: {e}", file=sys.stderr)
            return [0.0] * 7

    def _preprocess(self, obs: dict) -> dict:
        """Convert JSON obs to model input dict."""
        inputs = {}

        # RGB image
        if "rgb_static" in obs and obs["rgb_static"]:
            rgb = np.array(obs["rgb_static"], dtype=np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            # HWC → (1, 3, H, W)
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Select key based on model requirements
            inputs["pixel_values"] = rgb_t
            inputs["images"] = rgb_t

        # Language instruction
        language = obs.get("language", "")
        if language:
            inputs["instruction"] = language
            inputs["text"] = [language]

        # Proprio
        if "proprio" in obs and obs["proprio"] is not None:
            inputs["proprio"] = torch.tensor(
                obs["proprio"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        return inputs

    def reset(self):
        """Reset latent history."""
        self.latent_buffer.clear()
        if self.model is not None:
            if hasattr(self.model, "reset"):
                self.model.reset()
            if hasattr(self.model, "reset_history"):
                self.model.reset_history()


def main():
    parser = argparse.ArgumentParser(description="DreamZero Persistent Inference Server")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR, help="Model checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[DreamZero-Server] Starting (device={device})...", file=sys.stderr)
    sys.stderr.flush()

    server = DreamZeroInferenceServer(args.checkpoint, device)
    if server.is_ready:
        print(f"[DreamZero-Server] Ready with real model weights", file=sys.stderr)
    else:
        print(f"[DreamZero-Server] Running in stub mode (no model loaded)", file=sys.stderr)

    # Ready signal
    print(json.dumps({"status": "ready"}))
    sys.stdout.flush()

    # JSON Lines loop
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
            resp = {"error": str(e), "traceback": tb, "id": req.get("id", 0)}
            print(json.dumps(resp))
            sys.stdout.flush()

    print("[DreamZero-Server] Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
