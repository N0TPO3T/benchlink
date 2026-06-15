#!/usr/bin/env python3
"""
ManiFeelSimRunner — online evaluation based on IsaacGym simulation environment.

Start a Docker container (manifeel:l2-code), internally running manifeel_sim_server.py,
communicating via stdin/stdout JSON line protocol.

Data flow:
    Runner → CanonicalObs → Adapter.act() → standard_action (7,)
          → _from_canonical() → sim_action (6,)
          → env.step(action) inside Docker
          → Docker returns obs/reward/done
          → _to_canonical() → CanonicalObs → next round
"""

import subprocess
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


import numpy as np

from benchlink.base import BenchmarkRunner, ModelAdapter
from benchlink.schema import CanonicalObs


# Path to sim server script inside Docker
SIM_SERVER_DIR = str(Path(__file__).resolve().parent.parent)  # benchlink/
SIM_MOUNT = "/opt/manifeel_ef"
SIM_SERVER_CONTAINER = f"{SIM_MOUNT}/manifeel_sim_server.py"
RESP_PREFIX = "__SIM__"


class ManiFeelSimRunner(BenchmarkRunner):
    """ManiFeel IsaacGym simulation evaluation executor."""

    def __init__(self):
        super().__init__()
        self._proc: Optional[subprocess.Popen] = None
        self._resp_id: int = 0
        self._step_count: int = 0
        self._max_episode_length: int = 256

        # Current episode metrics
        self._episode_rewards = []
        self._cumulative_reward = 0.0

    def setup(self, config: dict) -> None:
        """Configure simulation environment.

        Required config fields:
            docker_image: Docker image name (default "manifeel:l2-code")
        Optional config fields:
            seed: random seed (default 42)
            max_episode_length: max steps per episode (default 256)
            num_envs: number of parallel environments (default 1)
            headless: headless rendering (default True)
        """
        self.docker_image = config.get("docker_image", "manifeel:l2-code")
        self.seed = config.get("seed", 42)
        self._max_episode_length = config.get("max_episode_length", 256)
        self.num_envs = config.get("num_envs", 1)
        self.headless = config.get("headless", True)

    def evaluate(
        self,
        model: ModelAdapter,
        n_episodes: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Closed-loop simulation evaluation.

        Per episode:
            env.reset() → model.reset()
            for step in range(max_episode_length):
                obs → model.act() → action → env.step(action)
                record reward, check done

        Returns:
            dict: {
                "success_rate": float,
                "mean_reward": float,
                "mean_episode_length": float,
                "n_episodes": int,
                "total_steps": int,
            }
        """
        self._start_server()

        successes = []
        episode_lengths = []

        for ep in range(n_episodes):
            model.reset()
            obs_dict = self._reset_env()
            self._cumulative_reward = 0.0
            step = 0

            for step in range(self._max_episode_length):
                # Standard observation dict → model inference
                canonical_obs = self._to_canonical(obs_dict)
                action = model.act(canonical_obs)  # (7,)

                # Standard action → sim native action (6-D delta)
                sim_action = self._from_canonical(action)  # (6,)

                # Environment step
                obs_dict, reward, done = self._step_env(sim_action)
                self._cumulative_reward += reward

                if done:
                    break

            successes.append(1.0 if done else 0.0)
            episode_lengths.append(step + 1)

        self._stop_server()

        return {
            "success_rate": float(np.mean(successes)),
            "mean_reward": float(np.mean(episode_lengths)),  # use reward as proxy if not available
            "mean_episode_length": float(np.mean(episode_lengths)),
            "n_episodes": n_episodes,
            "total_steps": int(sum(episode_lengths)),
        }

    # ── Internal: Docker communication ──

    def _start_server(self) -> None:
        """Start the sim server inside a Docker container."""

        cmd = [
            "docker", "run", "--rm", "-i", "--gpus", "all",
            "-v", f"{SIM_SERVER_DIR}:{SIM_MOUNT}",
            self.docker_image,
            "bash", "-c",
            f"apt-get install -y -qq python3-tk 2>&1 | tail -1 && "
            f"ln -sf /opt/manifeel-isaacgymenvs/assets /usr/local/lib/python3.8/dist-packages/assets && "
            f"/usr/bin/python3.8 {SIM_SERVER_CONTAINER}",
        ]

        self._log(f"Running: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Background thread to read stderr (prevent pipe buffer blockage)
        import threading
        _stderr_log = []

        def _read_stderr():
            while True:
                line = self._proc.stderr.readline()
                if not line:
                    break
                _stderr_log.append(line.rstrip())

        threading.Thread(target=_read_stderr, daemon=True).start()

        # Look for "Ready" in stderr
        self._log("Waiting for sim server...")
        start = time.time()
        while time.time() - start < 300:
            for line in list(_stderr_log):
                if "Ready" in line:
                    self._log("Sim server ready")
                    time.sleep(0.5)  # Ensure server enters stdin listening mode
                    return
            if self._proc.poll() is not None:
                self._log(f"  Process died with exit={self._proc.poll()}")
                break
            if len(_stderr_log) % 5 == 0 and len(_stderr_log) > 0:
                pass  # log throttled
            time.sleep(0.1)

        err_text = "\n".join(_stderr_log[-50:])
        raise RuntimeError(
            f"Sim server failed. Exit={self._proc.poll()}. "
            f"stderr (last 50 lines):\n{err_text[:3000]}"
        )

    def _stop_server(self) -> None:
        """Shut down the Docker container."""
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send_request(self, request: dict) -> dict:
        """Send request to sim server and get response."""
        self._resp_id += 1
        request["id"] = self._resp_id

        req_line = json.dumps(request) + "\n"
        self._proc.stdin.write(req_line)
        self._proc.stdin.flush()

        while True:
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                poll = self._proc.poll()
                raise RuntimeError(
                    f"Sim server closed connection (exit={poll}). "
                    f"Request was: {request.get('reset', False) or 'step'}"
                )
            resp_line = resp_line.strip()
            if resp_line.startswith(RESP_PREFIX):
                resp = json.loads(resp_line[len(RESP_PREFIX):])
                break

        if "error" in resp:
            raise RuntimeError(f"Sim error: {resp['error']}\n{resp.get('traceback', '')}")
        return resp

    def _reset_env(self) -> dict:
        """Reset simulation environment, return initial observation dict."""
        resp = self._send_request({"reset": True})
        self._log(f"Reset response keys: {list(resp.keys())}, obs={resp.get('obs', 'MISSING')}")
        self._step_count = 0
        if "obs" not in resp:
            raise KeyError(f"Reset response missing 'obs' key. Full: {resp}")
        return resp["obs"]

    def _step_env(self, action: np.ndarray) -> tuple:
        """Execute one simulation step.

        Args:
            action: (6,) np.ndarray [dx, dy, dz, drot_x, drot_y, drot_z]

        Returns:
            (obs_dict, reward, done)
        """
        action_list = action[:6].tolist()
        resp = self._send_request({"action": action_list})
        self._step_count += 1
        return resp["obs"], resp["reward"], resp["done"]

    # ── Format conversion ──

    def _to_canonical(self, sim_obs: dict) -> CanonicalObs:
        """IsaacGym named observation dict → CanonicalObs.

        P0: use ee_pos/ee_quat returned from simulation as proprio.
        Visual inputs use random noise (simulation cameras not enabled).
        """
        # Assemble proprio from named tensors (ee_pos + ee_quat)
        ee_pos = np.asarray(sim_obs.get("ee_pos", [0, 0, 0]), dtype=np.float32).flatten()
        ee_quat = np.asarray(sim_obs.get("ee_quat", [0, 0, 0, 1]), dtype=np.float32).flatten()
        proprio = np.concatenate([ee_pos, ee_quat])  # (7,)

        # P0: use random noise as image placeholder (simulation cameras not enabled)
        dummy_img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

        return CanonicalObs(
            rgb_static=dummy_img,
            rgb_gripper=dummy_img,
            depth=None,
            proprio=proprio,
            tactile_img=dummy_img,
            tactile_feat=np.random.randn(64).astype(np.float32),
            language="insert the USB connector into the port",
        )

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """Standard action (7,) → IsaacGym sim action (6,).

        Standard: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        IsaacGym: [dx, dy, dz, drot_x, drot_y, drot_z] (6-D task space impedance delta)
        """
        return action[:6].copy()

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[ManiFeelSim] {msg}", file=sys.stderr)
