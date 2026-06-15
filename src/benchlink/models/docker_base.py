#!/usr/bin/env python3
"""
DockerModelAdapter -- Docker container adapter base class.

Encapsulates common logic for three Docker adapters (OpenPi/Motus/DreamZero):
  - Container lifecycle (_ensure_container, _connect_existing_container)
  - Inference server subprocess management (_start_server, close)
  - JSON Lines protocol communication (_send_request)
  - Standard action boundary handling

Subclasses only need to implement:
  1. `_get_defaults()` -- Return default config dict
  2. `_build_request(obs)` -- Construct request dict from CanonicalObs
  3. `act(obs)` -- Single-step inference (optional override of default implementation)
"""

import json
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM


class DockerModelAdapter(ModelAdapter, ABC):
    """Docker container adapter base class.

    Subclass configuration items (set in _get_defaults()):
        container_name:          Docker container name
        image:                   Docker image name
        gpu_ids:                 GPU device ID
        server_script:           Inference server path inside container
        conda_env:               Conda environment name (optional)
        server_start_timeout:    Seconds to wait for ready signal
        action_low/action_high:  Action bounds 7-D list

    Request format:
        Subclass controls the request dict structure via _build_request(obs).
    """

    def __init__(self):
        super().__init__()
        defaults = self._get_defaults()

        self.container_name: str = defaults.get("container_name", "docker_server")
        self.image: str = defaults.get("image", "image:latest")
        self._server_proc: Optional[subprocess.Popen] = None
        self._server_ready: bool = False
        self._req_id: int = 0
        self._server_start_timeout: int = defaults.get("server_start_timeout", 120)
        self._conda_env: str = defaults.get("conda_env", "")

        # Action bounds
        self._action_low: np.ndarray = np.asarray(
            defaults.get("action_low", [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0]),
            dtype=np.float64,
        )
        self._action_high: np.ndarray = np.asarray(
            defaults.get("action_high", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]),
            dtype=np.float64,
        )

    # ── Subclass must implement ──

    @abstractmethod
    def _get_defaults(self) -> dict:
        """Return the default config dict for this adapter."""
        ...

    @abstractmethod
    def _build_request(self, obs: CanonicalObs) -> dict:
        """Convert CanonicalObs to inference request dict.

        The returned dict structure depends on the specific model's required JSON format.
        """
        ...

    # ── Public interface ──

    def load(self, checkpoint: str, config: dict) -> None:
        """Load model: connect/create container + start inference server.

        Args:
            checkpoint: Model path inside container (unused by some adapters)
            config: Config dict, overrides defaults from _get_defaults()
        """
        # Override defaults with config
        if "container_name" in config:
            self.container_name = config["container_name"]
        if "image" in config:
            self.image = config["image"]
        if "server_script" in config:
            self._server_script = config["server_script"]
        if "server_start_timeout" in config:
            self._server_start_timeout = config["server_start_timeout"]
        if "conda_env" in config:
            self._conda_env = config["conda_env"]
        if "action_low" in config:
            self._action_low = np.asarray(config["action_low"], dtype=np.float64)
        if "action_high" in config:
            self._action_high = np.asarray(config["action_high"], dtype=np.float64)

        self._server_script = config.get(
            "server_script",
            getattr(self, "_server_script", f"/workspace/{self.__class__.__name__.lower()}_inference_server.py")
        )

        use_existing = config.get("use_existing", True)
        start_container = config.get("start_container", True)

        if use_existing:
            self._connect_existing_container(config)
        elif start_container:
            self._ensure_container(config)
        else:
            raise ValueError(
                "Neither use_existing=True nor start_container=True. "
                "Set one to enable container connection."
            )

        self._start_server(config)

        self.config = config

    def reset(self) -> None:
        """Reset model state (send reset signal)."""
        if self._server_ready:
            self._req_id += 1
            self._send_request({"reset": True, "id": self._req_id, "_no_action": True})

    def close(self) -> None:
        """Close server connection and container."""
        if self._server_ready:
            try:
                self._send_request({"stop": True, "id": self._req_id, "_no_action": True})
            except Exception:
                pass
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.stdin.close()
                self._server_proc.stdout.close()
                self._server_proc.stderr.close()
            except Exception:
                pass
            self._server_proc = None
        self._server_ready = False

    # ── Internal: container management ──

    def _connect_existing_container(self, config: dict) -> None:
        """Connect to an existing container."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Container '{self.container_name}' not found. "
                f"Available image: {self.image}"
            )

        status = result.stdout.strip()
        if status != "running":
            print(f"[{self.__class__.__name__}] Starting container (status={status})...")
            subprocess.run(["docker", "start", self.container_name],
                           capture_output=True, check=True)

    def _ensure_container(self, config: dict) -> None:
        """Ensure the Docker container exists and is running."""
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={self.container_name}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        exists = self.container_name in result.stdout.strip()

        if not exists:
            gpu_ids = config.get("gpu_ids", "0")
            shm_size = config.get("shm_size", "16g")
            cmd = [
                "docker", "run", "-d", "--gpus", f"device={gpu_ids}",
                "--name", self.container_name,
                "--shm-size", shm_size,
                self.image,
                "tail", "-f", "/dev/null",
            ]
            print(f"[{self.__class__.__name__}] Creating container: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create container: {result.stderr.strip()}"
                )

        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status != "running":
            print(f"[{self.__class__.__name__}] Starting container (status={status})...")
            subprocess.run(["docker", "start", self.container_name],
                           capture_output=True, check=True)

    def _start_server(self, config: dict) -> None:
        """Start the persistent inference server process."""
        server_script = config.get("server_script", self._server_script)

        if self._conda_env:
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "bash", "-c",
                f"source /opt/conda/etc/profile.d/conda.sh && "
                f"conda activate {self._conda_env} && python {server_script}",
            ]
        else:
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "python", server_script,
            ]

        self._server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            deadline = time.time() + self._server_start_timeout
            while time.time() < deadline:
                line = self._server_proc.stdout.readline()
                if not line:
                    stderr = self._server_proc.stderr.read()
                    raise RuntimeError(
                        f"{self.__class__.__name__} server exited prematurely.\n"
                        f"stderr: {stderr}"
                    )
                try:
                    resp = json.loads(line.strip())
                    if resp.get("status") == "ready":
                        self._server_ready = True
                        print(f"[{self.__class__.__name__}] Server ready ({self.container_name})")
                        return
                except json.JSONDecodeError:
                    continue

            raise TimeoutError(
                f"{self.__class__.__name__} server did not become ready within "
                f"{self._server_start_timeout}s"
            )
        except Exception:
            self.close()
            raise

    def _send_request(self, request: dict) -> np.ndarray:
        """Send JSON request -> Read JSON response -> Return action vector."""
        if self._server_proc is None:
            raise RuntimeError("Server process not initialized")
        if self._server_proc.poll() is not None:
            raise RuntimeError("Inference server process has exited")

        line = json.dumps(request) + "\n"
        self._server_proc.stdin.write(line)
        self._server_proc.stdin.flush()

        resp_line = self._server_proc.stdout.readline()
        if not resp_line:
            stderr = self._server_proc.stderr.read()
            raise RuntimeError(
                f"{self.__class__.__name__} server connection lost.\n"
                f"stderr: {stderr}"
            )

        try:
            response = json.loads(resp_line.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse server response: {e}\nRaw: {resp_line}"
            )

        if "error" in response:
            raise RuntimeError(
                f"{self.__class__.__name__} server error: {response['error']}"
            )

        # reset/stop requests don't need action key
        if request.get("_no_action"):
            return np.zeros(STANDARD_ACTION_DIM, dtype=np.float64)

        raw_action = response.get("action", response.get("raw_action"))
        if raw_action is None:
            raise KeyError(f"Server response missing 'action' key: {response}")

        return np.asarray(raw_action, dtype=np.float64).flatten()

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to bounds."""
        return np.clip(action, self._action_low, self._action_high)
