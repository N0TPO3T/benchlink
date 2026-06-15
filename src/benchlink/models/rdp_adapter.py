#!/usr/bin/env python3
"""
RDP (Reactive Diffusion Policy) -> ModelAdapter.

RDP consists of two components:
  - AT (Action Tokenizer): image -> discrete token
  - LDP (Latent Diffusion Policy): token + state -> denoising -> action sequence

Native action space: [x, y, z, gripper_width] — TCP absolute position, no rotation (4-D)
Standard action space: [dx, dy, dz, droll, dpitch, dyaw, gripper] — delta pose (7-D)

Architecture:
  Launches a persistent inference server via subprocess on startup (rdp_inference_server.py),
  running in the conda rdp environment. Each subsequent act() communicates via stdin/stdout
  JSON line protocol. This way the model loads only once at startup, making subsequent
  inference fast.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "rdp_inference_server.py"


class RDPAdapter(ModelAdapter):
    """RDP adapter -- persistent inference server mode."""

    def __init__(self):
        super().__init__()
        self.conda_env: str = "rdp"
        self.rdp_repo: str = ""
        self.ldp_checkpoint: str = ""
        self.at_checkpoint: str = ""
        self._server_proc: Optional[subprocess.Popen] = None
        self._server_ready: bool = False
        self._req_id: int = 0
        self._prev_tcp: Optional[np.ndarray] = None

    def load(self, checkpoint: str, config: dict) -> None:
        """Configure RDP model paths (server starts lazily).

        Args:
            checkpoint: Path to LDP checkpoint
            config: Must include:
                rdp_repo:   RDP code repository path
                conda_env:  conda environment name (default "rdp")
                Optional:
                at_checkpoint: AT checkpoint path
        """
        self.ldp_checkpoint = str(Path(checkpoint).resolve())
        self.rdp_repo = str(Path(config["rdp_repo"]).resolve())
        self.conda_env = config.get("conda_env", "rdp")
        self.at_checkpoint = config.get("at_checkpoint", "")

        # Verify paths exist
        if not os.path.exists(self.ldp_checkpoint):
            raise FileNotFoundError(f"RDP LDP checkpoint not found: {self.ldp_checkpoint}")
        if not os.path.exists(self.rdp_repo):
            raise FileNotFoundError(f"RDP repo not found: {self.rdp_repo}")
        if self.at_checkpoint and not os.path.exists(self.at_checkpoint):
            raise FileNotFoundError(f"RDP AT checkpoint not found: {self.at_checkpoint}")

        self.device = config.get("device", "cuda")

    def act(self, obs: CanonicalObs) -> np.ndarray:
        """Single-step inference.

        Starts the persistent inference server on first call.
        """
        # -- Input validation --
        if obs.rgb_gripper is None or obs.proprio is None:
            raise ValueError("RDPAdapter requires obs.rgb_gripper and obs.proprio")

        # -- Lazily start the inference server --
        if not self._server_ready:
            self._start_server()

        # -- Extract proprio --
        proprio = obs.proprio
        tcp_pose = proprio[:3].tolist()
        gripper_width = float(proprio[3]) if len(proprio) > 3 else 0.0

        # -- Tactile -> GelSight embedding --
        tactile_emb = self._tactile_to_emb(obs.tactile_feat)

        # -- Build request --
        self._req_id += 1
        request = {
            "id": self._req_id,
            "obs": {
                "wrist_img": obs.rgb_gripper.tolist(),
                "tcp_pose": tcp_pose,
                "gripper_width": gripper_width,
                "tactile_emb": tactile_emb,  # already a list from _tactile_to_emb
            },
        }

        # -- Send request, read response --
        action_pred = self._server_request(request)  # [x, y, z, gripper_width]

        # -- Absolute TCP -> delta pose --
        delta_xyz = np.array(action_pred[:3]) - np.array(tcp_pose)
        delta_rot = np.zeros(3, dtype=np.float32)
        delta_gripper = float(action_pred[3]) - gripper_width

        action = np.array(
            [*delta_xyz.tolist(), *delta_rot.tolist(), delta_gripper],
            dtype=np.float64,
        )
        assert action.shape == (STANDARD_ACTION_DIM,)
        return action

    def reset(self) -> None:
        """Called at the start of each episode."""
        self._prev_tcp = None

    def close(self) -> None:
        """Shut down the inference server."""
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=10)
            except Exception:
                self._server_proc.kill()
            self._server_proc = None
            self._server_ready = False

    # -- Internal methods --

    def _start_server(self) -> None:
        """Start the persistent RDP inference server process."""
        server_py = str(SERVER_SCRIPT.resolve())
        cmd = [
            str(Path.home() / ".conda/envs" / self.conda_env / "bin/python"),
            server_py,
            "--ldp-checkpoint", self.ldp_checkpoint,
            "--rdp-repo", self.rdp_repo,
            "--device", self.device,
        ]
        if self.at_checkpoint:
            cmd.extend(["--at-checkpoint", self.at_checkpoint])

        self._log("Starting RDP inference server...")
        self._log(f"  Cmd: {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Wait for server ready (read "Ready" signal)
        ready_line = self._server_proc.stderr.readline()
        if "Ready" not in ready_line:
            # Try waiting longer
            import time
            time.sleep(2)
            # Check if still alive
            poll = self._server_proc.poll()
            if poll is not None:
                stderr_out = self._server_proc.stderr.read()
                raise RuntimeError(
                    f"RDP server died on startup (exit={poll}). "
                    f"stderr:\n{stderr_out[:2000]}"
                )
            # Not dead and not ready, keep waiting
            self._log(f"  Server stderr (initial): {ready_line.strip()}")

        self._server_ready = True
        self._log("RDP inference server ready")

    def _server_request(self, request: dict) -> list:
        """Send a request to the inference server and get the response."""
        if self._server_proc is None or self._server_proc.poll() is not None:
            raise RuntimeError("RDP inference server is not running")

        # Send request
        req_line = json.dumps(request) + "\n"
        self._server_proc.stdin.write(req_line)
        self._server_proc.stdin.flush()

        # Read response (skip model log lines, look for __RDP__ prefix)
        RESP_PREFIX = "__RDP__"
        while True:
            resp_line = self._server_proc.stdout.readline()
            if not resp_line:
                stderr_out = self._server_proc.stderr.read()
                raise RuntimeError(
                    f"RDP server closed connection.\n"
                    f"stderr:\n{stderr_out[:2000]}"
                )
            resp_line = resp_line.strip()
            if resp_line.startswith(RESP_PREFIX):
                resp = json.loads(resp_line[len(RESP_PREFIX):])
                break
            # Otherwise it is a model log line, ignore

        if "error" in resp:
            tb = resp.get("traceback", "N/A")
            raise RuntimeError(
                f"RDP inference error: {resp['error']}\n"
                f"Server traceback:\n{tb}"
            )

        return resp["action"]  # list of 4 floats

    @staticmethod
    def _tactile_to_emb(tactile_feat: Optional[np.ndarray]) -> list:
        """Tactile feature -> RDP GelSight embedding (15,)."""
        if tactile_feat is None:
            return [0.0] * 15
        emb = np.zeros(15, dtype=np.float32)
        n = min(len(tactile_feat), 15)
        emb[:n] = tactile_feat[:n]
        return emb.tolist()

    @staticmethod
    def _log(msg: str) -> None:
        """Log a message to stderr prefixed with [RDPAdapter]."""
        print(f"[RDPAdapter] {msg}", file=sys.stderr)
