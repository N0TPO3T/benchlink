#!/usr/bin/env python3
"""
Docker adapter end-to-end smoke test.

Usage:
    # Test a single adapter
    python deploy/test_docker_adapter.py openpi
    python deploy/test_docker_adapter.py motus
    python deploy/test_docker_adapter.py dreamzero

    # Test all
    python deploy/test_docker_adapter.py all

Each test:
    1. Verify container exists and is running
    2. Create adapter instance, load()
    3. Send dummy CanonicalObs → check action shape
    4. reset() → infer again
    5. close()
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchlink.schema import CanonicalObs
from benchlink.registry import get_model


TEST_IMAGE = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
TEST_LANGUAGE = "pick up the object"


def make_dummy_obs() -> dict:
    """Create a CanonicalObs for testing (as dict, will be converted by adapter)."""
    return {
        "rgb_static": TEST_IMAGE,
        "language": TEST_LANGUAGE,
        "proprio": np.zeros(8, dtype=np.float64),
    }


def test_adapter(name: str, config: dict) -> bool:
    """Test the full lifecycle of a Docker adapter.

    Returns:
        True if all tests pass
    """
    print(f"\n{'='*60}")
    print(f"  Testing adapter: {name}")
    print(f"{'='*60}")

    # 1. Check container
    container_name = config.get("container_name", f"{name}_server")
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [SKIP] Container '{container_name}' not found. Skipping.")
        print(f"         Create it: docker run -d --gpus device=0 --name {container_name} ...")
        return False

    status = result.stdout.strip()
    if status != "running":
        print(f"  [SKIP] Container '{container_name}' status={status}. Skipping.")
        return False

    print(f"  [OK] Container '{container_name}' is running (status={status})")

    # 2. Create adapter
    try:
        adapter_cls = get_model(name)
        adapter = adapter_cls()
    except KeyError:
        print(f"  [FAIL] Model '{name}' not registered in registry.")
        return False
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate adapter: {e}")
        return False
    print(f"  [OK] Instantiated {type(adapter).__name__}")

    # 3. load()
    try:
        adapter.load("", config)
        print(f"  [OK] load() completed, server ready={adapter._server_ready}")
    except Exception as e:
        print(f"  [FAIL] load() failed: {e}")
        return False

    # 4. act() with dummy obs
    try:
        obs_dict = make_dummy_obs()
        obs = CanonicalObs(**obs_dict)
        action = adapter.act(obs)
        assert isinstance(action, np.ndarray), f"action type={type(action)}"
        assert action.shape == (7,), f"action shape={action.shape}"
        assert action.dtype == np.float64, f"action dtype={action.dtype}"
        print(f"  [OK] act() → shape={action.shape}, dtype={action.dtype}")
        print(f"       action = [{', '.join(f'{v:.4f}' for v in action[:3])}, ...]")
    except Exception as e:
        print(f"  [FAIL] act() failed: {e}")
        adapter.close()
        return False

    # 5. reset() → act() again
    try:
        adapter.reset()
        obs_dict = make_dummy_obs()
        obs = CanonicalObs(**obs_dict)
        action2 = adapter.act(obs)
        assert action2.shape == (7,), f"After reset, action shape={action2.shape}"
        print(f"  [OK] reset() + act() → shape={action2.shape}")
    except Exception as e:
        print(f"  [FAIL] reset() → act() failed: {e}")
        adapter.close()
        return False

    # 6. close()
    try:
        adapter.close()
        assert adapter._server_proc is None or adapter._server_proc.poll() is not None, \
            "Server process still running after close()"
        print("  [OK] close() completed")
    except Exception as e:
        print(f"  [FAIL] close() failed: {e}")
        return False

    print(f"\n  ✅ {name} ALL TESTS PASSED")
    return True


def main():
    adapters_to_test = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    # Define configs
    adapter_configs = {
        "openpi": {
            "container_name": "openpi_server",
            "image": "openpi:pi05-libero-rollout-ok",
            "server_script": "/workspace/openpi_inference_server.py",
            "start_container": False,
            "use_existing": True,
        },
        "motus": {
            "container_name": "motus_server",
            "image": "motus:full",
            "server_script": "/workspace/motus_inference_server.py",
            "start_container": False,
            "use_existing": True,
        },
        "dreamzero": {
            "container_name": "dreamzero_final",
            "image": "dreamzero:int8-final",
            "server_script": "/workspace/dreamzero_inference_server.py",
            "start_container": False,
            "use_existing": True,
            "conda_env": "dreamzero",
        },
    }

    if "all" in adapters_to_test:
        to_test = list(adapter_configs.keys())
    else:
        to_test = [a for a in adapters_to_test if a in adapter_configs]
        unknown = [a for a in adapters_to_test if a not in adapter_configs]
        if unknown:
            print(f"Unknown adapters: {unknown}")
            print(f"Available: {list(adapter_configs.keys())}, all")

    # ── Run tests ──
    results = {}
    for name in to_test:
        passed = test_adapter(name, adapter_configs[name])
        results[name] = passed

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
