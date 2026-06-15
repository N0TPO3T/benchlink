#!/usr/bin/env python3
"""
RDP persistent inference server.

Runs in the conda rdp environment, loads the model once, handles inference requests via stdin/stdout.

Usage (server side):
    nohup /path/to/.conda/envs/rdp/bin/python rdp_inference_server.py \
        --ldp-checkpoint /path/to/ldp/latest.ckpt \
        --at-checkpoint /path/to/at/latest.ckpt \
        --rdp-repo /path/to/RDP \
        > server.log 2>&1 &

Protocol (JSON Lines):
    stdin  ←  One request per line:  {"obs": {...}, "id": 1}
    stdout →  One response per line:  {"action": [x,y,z,g], "id": 1}

Supported obs keys:
    - wrist_img:    list of floats, shape (H, W, 3), or shape (n_steps, H, W, 3)
    - tcp_pose:     list of floats, shape (3,)
    - gripper_width: float
    - tactile_emb:  list of floats, shape (15,)
"""

import sys
import json
import argparse
import os
import traceback
from pathlib import Path

import dill
import torch
import numpy as np
from omegaconf import OmegaConf


TARGET_MAP = {
    "diffusion_policy.model.vq_bet.vqvae.vqvae.VqVae":
        "reactive_diffusion_policy.model.vae.model.VAE",
    "diffusion_policy.policy.latent_diffusion_unet_image_policy.LatentDiffusionUnetImagePolicy":
        "reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy.LatentDiffusionUnetImagePolicy",
    "diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder":
        "reactive_diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder",
}


def _fix_targets(cfg):
    if OmegaConf.is_dict(cfg):
        if "_target_" in cfg and isinstance(cfg._target_, str) and cfg._target_ in TARGET_MAP:
            cfg._target_ = TARGET_MAP[cfg._target_]
        for k in cfg:
            _fix_targets(cfg[k])
    elif OmegaConf.is_list(cfg):
        for item in cfg:
            _fix_targets(item)


def build_model(ldp_checkpoint: str, at_checkpoint: str, rdp_repo: str, device: torch.device):
    """加载 RDP 模型（LDP + AT），返回 model 对象。"""
    sys.path.insert(0, rdp_repo)
    # RDP depends on some diffusion_policy modules
    dp_dir = str(Path(rdp_repo).parent / "dp" / "diffusion_policy")
    if os.path.exists(dp_dir):
        sys.path.insert(0, dp_dir)

    # 1. Load LDP checkpoint
    print(f"[RDP-Server] Loading LDP: {ldp_checkpoint}", file=sys.stderr)
    ldp_ckpt = torch.load(ldp_checkpoint, pickle_module=dill, map_location="cpu")
    _fix_targets(ldp_ckpt["cfg"])

    # 2. Fix policy config (rename vqvae → at)
    pol = ldp_ckpt["cfg"].policy
    OmegaConf.set_struct(pol, False)
    if "vqvae" in pol:
        pol["at"] = pol["vqvae"]
        OmegaConf.set_struct(pol["at"], False)
        if "load_dir" in pol["at"]:
            pol["at"].load_dir = None
        pol.pop("vqvae")

    from reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace import \
        TrainDiffusionUnetImageWorkspace

    ldp_ws = TrainDiffusionUnetImageWorkspace(ldp_ckpt["cfg"])
    ldp_ws.load_checkpoint(path=ldp_checkpoint)
    model = ldp_ws.model
    model.eval().to(device)
    torch.cuda.synchronize()
    print(f"[RDP-Server] LDP loaded: {type(model).__name__}", file=sys.stderr)

    # 3. Set identity normalizer (P0: pass-through)
    class IdentityNormalizer(torch.nn.Module):
        def normalize(self, x):
            if isinstance(x, dict):
                return {k: self.normalize(v) for k, v in x.items()}
            return x.float() if not x.is_floating_point() else x

        def unnormalize(self, x):
            return x

        def __getitem__(self, key):
            return self

    id_norm = IdentityNormalizer()
    model.normalizer = id_norm
    if hasattr(model, "at") and hasattr(model.at, "normalizer"):
        model.at.normalizer = id_norm

    # 4. Determine n_obs_steps and downsample_ratio from config
    n_obs_steps = ldp_ckpt["cfg"].get("n_obs_steps", 2)
    downsample_ratio = ldp_ckpt["cfg"].get("dataset_obs_temporal_downsample_ratio", 2)
    print(f"[RDP-Server] n_obs_steps={n_obs_steps}, downsample_ratio={downsample_ratio}, ready", file=sys.stderr)
    return model, n_obs_steps, downsample_ratio


def request_to_obs(req: dict, n_obs_steps: int, device: torch.device) -> dict:
    """Convert JSON request to model input obs_dict."""
    # Process image: single frame → repeat to n_obs_steps frames
    wrist_img = np.array(req["obs"]["wrist_img"], dtype=np.float32)  # (H, W, 3)

    # Resize to model's expected input size (240, 320) using numpy
    target_h, target_w = 240, 320
    if wrist_img.ndim == 3:
        if wrist_img.shape[:2] != (target_h, target_w):
            wrist_img = _resize_np(wrist_img, target_h, target_w)
        wrist_img = np.stack([wrist_img] * n_obs_steps, axis=0)
    elif wrist_img.ndim == 4:
        frames = [_resize_np(wrist_img[i], target_h, target_w)
                  if wrist_img[i].shape[:2] != (target_h, target_w)
                  else wrist_img[i]
                  for i in range(wrist_img.shape[0])]
        wrist_img = np.stack(frames, axis=0)

    wrist_img = torch.from_numpy(wrist_img).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    # wrist_img shape: (1, n_obs_steps, 3, H, W)

    tcp = torch.tensor(req["obs"]["tcp_pose"], dtype=torch.float32, device=device)
    tcp = tcp.unsqueeze(0).unsqueeze(1).expand(-1, n_obs_steps, -1)
    # tcp shape: (1, n_obs_steps, 3)

    gripper = torch.tensor([[req["obs"]["gripper_width"]]], dtype=torch.float32, device=device)
    gripper = gripper.unsqueeze(1).expand(-1, n_obs_steps, -1)
    # gripper shape: (1, n_obs_steps, 1)

    tactile_emb = torch.tensor(req["obs"]["tactile_emb"], dtype=torch.float32, device=device)
    tactile_emb = tactile_emb.unsqueeze(0).unsqueeze(1).expand(-1, n_obs_steps, -1)
    # tactile_emb shape: (1, n_obs_steps, 15)

    obs = {
        "left_wrist_img": wrist_img,
        "left_robot_tcp_pose": tcp,
        "left_robot_gripper_width": gripper,
        "left_gripper1_marker_offset_emb": tactile_emb,
    }
    # extended_obs: AT decoder needs additional temporal conditioning
    extended_obs = {
        "left_gripper1_marker_offset_emb": tactile_emb,
    }
    return obs, extended_obs


@torch.no_grad()
def infer(model, obs_dict: dict, extended_obs_dict: dict, downsample_ratio: int) -> list:
    """Run inference, return action list [x, y, z, gripper_width]."""
    out = model.predict_action(
        obs_dict,
        dataset_obs_temporal_downsample_ratio=downsample_ratio,
        extended_obs_dict=extended_obs_dict,
    )

    if isinstance(out, dict):
        # action_pred is the actual prediction, action is sometimes empty (shape [1, 0, 4])
        if "action_pred" in out and out["action_pred"].shape[1] > 0:
            action = out["action_pred"]
        else:
            action = out.get("action", out)
    else:
        action = out

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    # action shape: (1, action_horizon, 4) or (1, 4)
    # rdp_check output: action_pred=(1, 2, 4), action=(1, 0, 4)
    if action.ndim == 3 and action.shape[1] > 0:
        action = action[0, 0]  # Take first prediction step
    elif action.ndim == 3:
        action = action[0, -1]  # Take last prediction step
    elif action.ndim == 2:
        action = action[0]

    return action.tolist()


def _resize_np(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """NumPy bilinear resize for (H, W, 3) images."""
    h, w = img.shape[:2]
    rows = np.linspace(0, h - 1, target_h)
    cols = np.linspace(0, w - 1, target_w)
    # Simple bilinear interpolation
    out = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    for r in range(target_h):
        r_orig = rows[r]
        r0, r1 = int(np.floor(r_orig)), min(int(np.ceil(r_orig)), h - 1)
        for c in range(target_w):
            c_orig = cols[c]
            c0, c1 = int(np.floor(c_orig)), min(int(np.ceil(c_orig)), w - 1)
            # Bilinear weights
            ry, cx = r_orig - r0, c_orig - c0
            out[r, c] = (1 - ry) * (1 - cx) * img[r0, c0] \
                      + ry * (1 - cx) * img[r1, c0] \
                      + (1 - ry) * cx * img[r0, c1] \
                      + ry * cx * img[r1, c1]
    return out


def main():
    parser = argparse.ArgumentParser(description="RDP Persistent Inference Server")
    parser.add_argument("--ldp-checkpoint", required=True)
    parser.add_argument("--at-checkpoint", default="")
    parser.add_argument("--rdp-repo", default="/path/to/reactive_diffusion_policy-main")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, n_obs_steps, downsample_ratio = build_model(
        args.ldp_checkpoint, args.at_checkpoint, args.rdp_repo, device
    )

    print("[RDP-Server] Ready, waiting for input...", file=sys.stderr)
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            obs_dict, extended_obs = request_to_obs(req, n_obs_steps, device)
            # Inference (no redirect needed; model prints # separators after inference)
            action = infer(model, obs_dict, extended_obs, downsample_ratio)
            resp = {"action": action, "id": req.get("id", 0)}
            sys.stdout.write("__RDP__" + json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            tb = traceback.format_exc()
            resp = {"error": str(e), "traceback": tb, "id": req.get("id", 0)}
            sys.stdout.write("__RDP__" + json.dumps(resp) + "\n")
            sys.stdout.flush()

    print("[RDP-Server] Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
