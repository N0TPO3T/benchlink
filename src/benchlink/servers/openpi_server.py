#!/usr/bin/env python3
"""
OpenPI (pi0) persistent inference server.

Runs inside the openpi Docker container, communicates via stdin/stdout JSON Lines protocol.

Usage (inside container):
    python /workspace/openpi_inference_server.py [--checkpoint ...]

Protocol (JSON Lines):
    stdin  ←  {"obs": {"rgb_static": list, "language": str, ...}, "id": N}
    stdout →  {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper], "id": N}
    stdout →  {"status": "ready"}                         # Startup ready signal
    stdin  ←  {"reset": true, "id": N}                    # Reset history
    stdin  ←  {"stop": true, "id": N}                     # Shut down server

Supported obs keys:
    - rgb_static:  (H, W, 3) numpy list → preprocess → tensor
    - language:    Task description string
    - proprio:     (N,) robot state (optional)
"""

import json
import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch


def load_pi0_model(checkpoint_path: str, device: torch.device):
    """Load pi0 model.

    Loads the model based on the actually installed openpi package. Supports two modes:
      1. octopi package (legacy)
      2. pi0 native format (new)
    """
    model = None
    load_error = None

    # Try loading via octopi
    try:
        from octopi.models import Pi0Model
        print("[OpenPI-Server] Using octopi.Pi0Model", file=sys.stderr)
        model = Pi0Model.load_from_checkpoint(checkpoint_path) if checkpoint_path else Pi0Model()
        model.to(device).eval()
        return model
    except ImportError as e:
        load_error = f"octopi not available: {e}"
    except Exception as e:
        load_error = f"octopi load failed: {e}"

    # Try loading via pi0 package
    try:
        import pi0
        print("[OpenPI-Server] Using pi0.load_model", file=sys.stderr)
        model = pi0.load_model(checkpoint_path or "pi0-base")
        model.to(device).eval()
        return model
    except ImportError as e:
        load_error = f"{load_error}; pi0 not available: {e}"
    except Exception as e:
        load_error = f"{load_error}; pi0 load failed: {e}"

    # Fallback: load directly with torch.load (for .pt/.pth)
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            print(f"[OpenPI-Server] Loading raw checkpoint: {checkpoint_path}", file=sys.stderr)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                # Try to resolve as model class
                model_cls = ckpt.get("model_class", None)
                if model_cls and model_cls in globals():
                    model = globals()[model_cls]()
                else:
                    model = ckpt  # Return state_dict directly
            else:
                model = ckpt
            model = model.to(device).eval() if hasattr(model, "eval") else model
            return model
        except Exception as e:
            load_error = f"{load_error}; raw load failed: {e}"

    raise RuntimeError(
        f"Cannot load pi0 model. Tried octopi, pi0, and raw torch.load.\n"
        f"Last error: {load_error}\n"
        f"Checkpoint: {checkpoint_path}"
    )


def preprocess_obs(obs: dict, device: torch.device) -> dict:
    """Convert JSON obs to model input tensor dict."""
    model_inputs = {}

    # RGB image: (H, W, 3) → (1, 3, H, W)
    if "rgb_static" in obs:
        rgb = np.array(obs["rgb_static"], dtype=np.float32)
        # Normalize to [0, 1] if in [0, 255]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # HWC → CHW
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        model_inputs["image"] = rgb

    # Language instruction
    if "language" in obs and obs["language"]:
        model_inputs["instruction"] = obs["language"]

    # Proprio
    if "proprio" in obs:
        model_inputs["proprio"] = torch.tensor(
            obs["proprio"], dtype=torch.float32, device=device
        ).unsqueeze(0)

    return model_inputs


@torch.no_grad()
def infer(model, model_inputs: dict, device: torch.device) -> list:
    """Run inference, return 7-D action [dx, dy, dz, droll, dpitch, dyaw, gripper]."""
    action = None

    # Try octopi API: model.act(image, instruction)
    if hasattr(model, "act") and callable(model.act):
        try:
            img = model_inputs.get("image")
            instr = model_inputs.get("instruction", "")
            if img is not None:
                raw = model.act(image=img, instruction=[instr])
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy()
                # raw shape: (1, action_dim) or (action_dim,)
                action = np.asarray(raw, dtype=np.float64).flatten()
                if action.size >= 7:
                    return action[:7].tolist()
        except Exception as e:
            print(f"[OpenPI-Server] model.act() failed: {e}", file=sys.stderr)

    # Try pi0 API: model.forward(obs) → action
    if action is None:
        try:
            raw = model(model_inputs)
            if isinstance(raw, torch.Tensor):
                raw = raw.cpu().numpy()
            elif isinstance(raw, dict):
                raw = raw.get("action", raw.get("actions", list(raw.values())[0]))
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy()
            action = np.asarray(raw, dtype=np.float64).flatten()
            if action.size >= 7:
                return action[:7].tolist()
        except Exception as e:
            print(f"[OpenPI-Server] model.forward() failed: {e}", file=sys.stderr)

    # Ultimate fallback: return zero actions
    print("[OpenPI-Server] WARNING: inference failed, returning zeros", file=sys.stderr)
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def main():
    parser = argparse.ArgumentParser(description="OpenPI (pi0) Persistent Inference Server")
    parser.add_argument("--checkpoint", default="", help="Model checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ──
    print(f"[OpenPI-Server] Loading model (device={device})...", file=sys.stderr)
    sys.stderr.flush()
    model = load_pi0_model(args.checkpoint, device)
    print("[OpenPI-Server] Model loaded, ready for requests.", file=sys.stderr)
    sys.stderr.flush()

    # ── Ready signal (first line of stdout must be status: ready) ──
    print(json.dumps({"status": "ready"}))
    sys.stdout.flush()

    # ── JSON Lines loop ──
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)

            # Stop command
            if req.get("stop"):
                break

            # Reset command (pi0 is usually stateless; clear history if any)
            if req.get("reset"):
                if hasattr(model, "reset_history"):
                    model.reset_history()
                resp = {"status": "ok", "id": req.get("id", 0)}
                print(json.dumps(resp))
                sys.stdout.flush()
                continue

            # Inference request
            obs = req.get("obs", req)
            model_inputs = preprocess_obs(obs, device)
            action = infer(model, model_inputs, device)

            resp = {"action": action, "id": req.get("id", 0)}
            print(json.dumps(resp))
            sys.stdout.flush()

        except Exception as e:
            tb = traceback.format_exc()
            resp = {"error": str(e), "traceback": tb, "id": req.get("id", 0)}
            print(json.dumps(resp))
            sys.stdout.flush()

    print("[OpenPI-Server] Shutting down.", file=sys.stderr)
    sys.stderr.flush()


if __name__ == "__main__":
    main()
