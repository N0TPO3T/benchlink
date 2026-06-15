#!/usr/bin/env python3
"""
ManiFeel IsaacGym Simulation Server.
Wraps the TacSLTaskInsertion environment inside Docker, stdin/stdout JSON line protocol.

Protocol:
    Input:  {"action": [6 floats], "reset": false}
    Output:  {"obs": {...}, "reward": 0.0, "done": false, "step": 0}
"""

import sys
import json
import os
import traceback

ISAACGYM_DIR = "/opt/isaacgym/python"
ISAACGYMENVS_DIR = "/usr/local/lib/python3.8/dist-packages/isaacgymenvs"
sys.path.insert(0, ISAACGYM_DIR)
sys.path.insert(0, ISAACGYMENVS_DIR)
os.environ["HYDRA_FULL_ERROR"] = "1"

import isaacgym  # noqa: F401, E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import hydra  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def create_env(seed=42, num_envs=1, headless=True):
    """Create TacSLTaskInsertion environment."""
    cfg_dir = f"{ISAACGYMENVS_DIR}/cfg"
    asset_dir = "/opt/manifeel-isaacgymenvs/assets/tacsl/yaml"

    # Compose all config in a single Hydra session
    with initialize_config_dir(config_dir=cfg_dir):
        # Register schema
        from isaacgymenvs.tasks.factory.factory_schema_config_env import (
            FactorySchemaConfigEnv,
        )
        from isaacgymenvs.tasks.factory.factory_schema_config_base import (
            FactorySchemaConfigBase,
        )
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        # Main config
        main_cfg = compose(
            config_name="config",
            overrides=[
                "task=TacSLTaskInsertion",
                "train=TacSLTaskInsertionPPO_LSTM_dict",
                f"num_envs={num_envs}",
                "headless=True",
            ],
        )

        # Env config + strip task nesting
        env_cfg = compose(
            config_name="task/TacSLEnvInsertion.yaml",
            overrides=["task.env.desired_subassemblies=[round_peg_hole_16mm]"],
        )["task"]

        # Base config + strip task nesting
        cfg_base = compose(config_name="task/TacSLBase.yaml")["task"]

    # Load asset YAML directly with OmegaConf (Hydra compose does not support ../ paths)
    def _load_yaml(path):
        p = os.path.join(asset_dir, path)
        if os.path.exists(p):
            return OmegaConf.load(p)
        # Fallback
        p2 = f"/opt/manifeel-isaacgymenvs/assets/tacsl/yaml/{path}"
        if os.path.exists(p2):
            return OmegaConf.load(p2)
        raise FileNotFoundError(f"Cannot load asset: {path}")

    asset_info = _load_yaml("industreal_asset_info_pegs.yaml")
    asset_info_franka = _load_yaml("tacsl_asset_info_franka_table.yaml")

    # Patch all _get_yaml_params methods (skip hydra.compose calls)
    from isaacgymenvs.tasks.tacsl.tacsl_env_insertion import TacSLEnvInsertion
    from isaacgymenvs.tasks.tacsl.tacsl_base import TacSLBase

    def _patch_env_params(self):
        self.cfg_env = env_cfg
        self.asset_info_insertion = asset_info

    def _patch_base_params(self):
        self.cfg_base = cfg_base
        self.asset_info_franka_table = asset_info_franka

    TacSLEnvInsertion._get_env_yaml_params = _patch_env_params
    TacSLBase._get_base_yaml_params = _patch_base_params

    # Create environment
    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

    sim_device = "cuda:0"
    rl_device = "cuda:0"
    task_config = OmegaConf.to_container(main_cfg.task, resolve=True)

    create_fn = get_rlgames_env_creator(
        seed=seed,
        task_config=task_config,
        task_name="TacSLTaskInsertion",
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=0,
        headless=headless,
        force_render=False,
    )
    env = create_fn()
    print(f"[SERVER] Env created. Action space: {env.action_space}", file=sys.stderr)

    # Post-processing: ensure missing attributes have defaults
    if not hasattr(env, 'socket_obs_noise'):
        env.socket_obs_noise = torch.zeros((num_envs, 3), device='cuda:0')
        print("[SERVER] Initialized socket_obs_noise=0", file=sys.stderr)

    return env


def obs_to_dict(obs):
    result = {}
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] > 0:
                # Take first env's tensor, flatten to list
                arr = v[0].cpu().flatten().tolist()
                if len(arr) <= 50:
                    result[k] = arr
                else:
                    result[k] = f"tensor{list(v.shape)}"
            elif isinstance(v, np.ndarray) and v.shape[0] > 0:
                arr = v[0].flatten().tolist()
                if len(arr) <= 50:
                    result[k] = arr
            elif isinstance(v, (int, float, bool, str)):
                result[k] = v
    return result


def main():
    # Suppress warnings and noisy logging before imports
    import logging
    logging.disable(logging.CRITICAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')

    # Save real stdout fd for __SIM__ responses, then redirect stdout→stderr
    _real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    real_stdout = os.fdopen(_real_stdout_fd, "w")

    print("[SERVER] Starting ManiFeel sim server...", file=sys.stderr)
    sys.stderr.flush()

    env = create_env(seed=42, num_envs=1, headless=True)
    obs = env.reset()

    print("[SERVER] Ready.", file=sys.stderr)
    sys.stderr.flush()

    step_count = 0
    resp_id = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp_id += 1
            if req.get("reset", False):
                obs = env.reset()
                step_count = 0
                resp = {"id": resp_id, "obs": obs_to_dict(obs), "reward": 0.0, "done": False, "step": 0}
                real_stdout.write("__SIM__" + json.dumps(resp) + "\n")
                real_stdout.flush()
                continue
            action_data = req["action"]
            action_tensor = torch.tensor([action_data], dtype=torch.float32, device="cuda:0")
            obs, reward, done, info = env.step(action_tensor)
            step_count += 1
            resp = {
                "id": resp_id,
                "obs": obs_to_dict(obs),
                "reward": float(reward[0].item()) if hasattr(reward, "item") else float(reward),
                "done": bool(done[0].item()) if hasattr(done, "item") else bool(done),
                "step": step_count,
            }
            real_stdout.write("__SIM__" + json.dumps(resp) + "\n")
            real_stdout.flush()
        except Exception as e:
            tb = traceback.format_exc()
            real_stdout.write("__SIM__" + json.dumps({"id": resp_id, "error": str(e), "traceback": tb}) + "\n")
            real_stdout.flush()

    try:
        env.destroy()
    except AttributeError:
        pass
    print("[SERVER] Shutdown.", file=sys.stderr)


if __name__ == "__main__":
    main()
