#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 minimal interface verification — random pairing of baseline × benchmark.

Test method:
  1. Use MockAdapter to simulate baseline (outputs standard actions or features randomly based on config)
  2. Use MockRunner to simulate benchmark
  3. Verify complete interface data flow: registration → setup → act/encode → evaluate
  4. Run two lines: action line (fastwam → libero) + tactile line (anytouch → anytouch_probe)

This script does not depend on any checkpoint / simulation environment / remote server.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

# ── Registration test (verify all components are registered) ──
def test_registration():
    print("=" * 60)
    print("[1/4] Registration Test")
    print("=" * 60)

    from benchlink.registry import (
        list_models, list_tactile_models, list_benchmarks,
    )

    # Verify registered classes can be imported successfully
    from benchlink.models.fastwam_adapter import FastWAMAdapter  # noqa: F401
    from benchlink.models.dp_adapter import DPAdapter  # noqa: F401
    from benchlink.models.anytouch_adapter import AnyTouchAdapter  # noqa: F401
    from benchlink.models.sparsh_adapter import SparshAdapter  # noqa: F401
    from benchlink.models.t3_adapter import T3Adapter  # noqa: F401
    from benchlink.benchmarks.libero_runner import LiberoRunner  # noqa: F401
    from benchlink.benchmarks.anytouch_probe_runner import AnyTouchProbeRunner  # noqa: F401
    from benchlink.benchmarks.maniskill_runner import ManiSkillRunner  # noqa: F401
    from benchlink.benchmarks.robotwin_runner import RoboTwinRunner  # noqa: F401
    from benchlink.benchmarks.droid_sim_runner import DroidSimRunner  # noqa: F401
    from benchlink.models.vla_touch_adapter import VLA_TouchAdapter  # noqa: F401
    from benchlink.models.unitac_ecf_adapter import UniTac_ECFAdapter  # noqa: F401
    from benchlink.models.openpi_adapter import OpenPiAdapter  # noqa: F401
    from benchlink.models.motus_adapter import MotusAdapter  # noqa: F401
    from benchlink.models.dreamzero_adapter import DreamZeroAdapter  # noqa: F401
    from benchlink.benchmarks.unitac_ecf_runner import UniTacECFRunner  # noqa: F401

    models = list_models()
    tactile_models = list_tactile_models()
    benches = list_benchmarks()

    print(f"  Models (Action):     {models}")
    print(f"  Tactile (Tactile):   {tactile_models}")
    print(f"  Benchmarks:          {benches}")

    assert "fastwam" in models, "FastWAMAdapter not registered"
    assert "dp" in models, "DPAdapter not registered"
    assert "rdp" in models, "RDPAdapter not registered"
    assert "openpi" in models, "OpenPiAdapter not registered"
    assert "motus" in models, "MotusAdapter not registered"
    assert "dreamzero" in models, "DreamZeroAdapter not registered"
    assert "vla_touch" in models, "VLA_TouchAdapter (action) not registered"
    assert "anytouch" in tactile_models, "AnyTouchAdapter not registered"
    assert "sparsh" in tactile_models, "SparshAdapter not registered"
    assert "t3" in tactile_models, "T3Adapter not registered"
    assert "vla_touch" in tactile_models, "VLA_TouchAdapter (tactile) not registered"
    assert "unitac_ecf" in tactile_models, "UniTac_ECFAdapter not registered"
    assert "libero" in benches, "LiberoRunner not registered"
    assert "anytouch_probe" in benches, "AnyTouchProbeRunner not registered"
    assert "maniskill" in benches, "ManiSkillRunner not registered"
    assert "robotwin" in benches, "RoboTwinRunner not registered"
    assert "droid_sim" in benches, "DroidSimRunner not registered"
    assert "unitac_ecf" in benches, "UniTacECFRunner not registered"

    # Verify registered classes can be imported successfully

    print("  ✅ All components registered correctly, imports successful")


# ── Interface signature test ──
def test_interface_signatures():
    print("\n" + "=" * 60)
    print("[2/4] Interface Signature Test")
    print("=" * 60)

    from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM

    # 1. CanonicalObs construction
    obs = CanonicalObs(
        rgb_static=np.zeros((256, 256, 3), dtype=np.uint8),
        rgb_gripper=np.zeros((128, 128, 3), dtype=np.uint8),
        proprio=np.zeros(7, dtype=np.float32),
        language="test task",
    )
    assert obs.rgb_static.shape == (256, 256, 3)
    assert obs.proprio.shape == (7,)
    print("  ✅ CanonicalObs construction correct")

    # 2. Standard action dimension
    assert STANDARD_ACTION_DIM == 7
    print(f"  ✅ STANDARD_ACTION_DIM = {STANDARD_ACTION_DIM}")

    # 3. FastWAMAdapter interface
    from benchlink.models.fastwam_adapter import FastWAMAdapter
    model = FastWAMAdapter()
    assert hasattr(model, "load"), "Missing load()"
    assert hasattr(model, "act"), "Missing act()"
    assert hasattr(model, "reset"), "Missing reset()"
    print("  ✅ FastWAMAdapter: load() / act() / reset() interface complete")

    # 4. AnyTouchAdapter interface
    from benchlink.models.anytouch_adapter import AnyTouchAdapter
    tactile = AnyTouchAdapter()
    assert hasattr(tactile, "load"), "Missing load()"
    assert hasattr(tactile, "encode"), "Missing encode()"
    print("  ✅ AnyTouchAdapter: load() / encode() interface complete")

    # 5. LiberoRunner interface
    from benchlink.benchmarks.libero_runner import LiberoRunner
    runner = LiberoRunner()
    assert hasattr(runner, "setup"), "Missing setup()"
    assert hasattr(runner, "evaluate"), "Missing evaluate()"
    assert hasattr(runner, "_to_canonical"), "Missing _to_canonical()"
    assert hasattr(runner, "_from_canonical"), "Missing _from_canonical()"
    print("  ✅ LiberoRunner: setup() / evaluate() / _to_canonical() / _from_canonical() interface complete")

    # 6. AnyTouchProbeRunner interface
    from benchlink.benchmarks.anytouch_probe_runner import AnyTouchProbeRunner
    probe = AnyTouchProbeRunner()
    assert hasattr(probe, "setup"), "Missing setup()"
    assert hasattr(probe, "evaluate"), "Missing evaluate()"
    print("  ✅ AnyTouchProbeRunner: setup() / evaluate() interface complete")

    # 7. ManiSkillRunner interface
    from benchlink.benchmarks.maniskill_runner import ManiSkillRunner
    ms = ManiSkillRunner()
    assert hasattr(ms, "setup"), "Missing setup()"
    assert hasattr(ms, "evaluate"), "Missing evaluate()"
    assert hasattr(ms, "_to_canonical"), "Missing _to_canonical()"
    assert hasattr(ms, "_from_canonical"), "Missing _from_canonical()"
    print("  ✅ ManiSkillRunner: setup() / evaluate() / _to_canonical() / _from_canonical() interface complete")

    # 8. RoboTwinRunner interface
    from benchlink.benchmarks.robotwin_runner import RoboTwinRunner
    rt = RoboTwinRunner()
    assert hasattr(rt, "setup"), "Missing setup()"
    assert hasattr(rt, "evaluate"), "Missing evaluate()"
    assert hasattr(rt, "_to_canonical"), "Missing _to_canonical()"
    assert hasattr(rt, "_from_canonical"), "Missing _from_canonical()"
    assert hasattr(rt, "close"), "Missing close()"
    print("  ✅ RoboTwinRunner: setup() / evaluate() / _to_canonical() / _from_canonical() / close() interface complete")

    # 9. DroidSimRunner interface
    from benchlink.benchmarks.droid_sim_runner import DroidSimRunner
    ds = DroidSimRunner()
    assert hasattr(ds, "setup"), "Missing setup()"
    assert hasattr(ds, "evaluate"), "Missing evaluate()"
    assert hasattr(ds, "_to_canonical"), "Missing _to_canonical()"
    assert hasattr(ds, "_from_canonical"), "Missing _from_canonical()"
    print("  ✅ DroidSimRunner: setup() / evaluate() / _to_canonical() / _from_canonical() interface complete")

    # 10. DPAdapter interface
    from benchlink.models.dp_adapter import DPAdapter
    dp = DPAdapter()
    assert hasattr(dp, "load"), "Missing load()"
    assert hasattr(dp, "act"), "Missing act()"
    assert hasattr(dp, "reset"), "Missing reset()"
    print("  ✅ DPAdapter: load() / act() / reset() interface complete")

    # 11. SparshAdapter interface
    from benchlink.models.sparsh_adapter import SparshAdapter
    sparsh = SparshAdapter()
    assert hasattr(sparsh, "load"), "Missing load()"
    assert hasattr(sparsh, "encode"), "Missing encode()"
    print("  ✅ SparshAdapter: load() / encode() interface complete")

    # 12. T3Adapter interface
    from benchlink.models.t3_adapter import T3Adapter
    t3 = T3Adapter()
    assert hasattr(t3, "load"), "Missing load()"
    assert hasattr(t3, "encode"), "Missing encode()"
    print("  ✅ T3Adapter: load() / encode() interface complete")

    # 13. VLA_TouchAdapter interface (dual interface)
    from benchlink.models.vla_touch_adapter import VLA_TouchAdapter
    vla = VLA_TouchAdapter()
    # ModelAdapter interface
    assert hasattr(vla, "load"), "Missing load()"
    assert hasattr(vla, "act"), "Missing act()"
    assert hasattr(vla, "reset"), "Missing reset()"
    # TactileAdapter interface
    assert hasattr(vla, "encode"), "Missing encode()"
    print("  ✅ VLA_TouchAdapter: load() / act() / reset() / encode() interface complete")

    # 14. UniTac_ECFAdapter interface
    from benchlink.models.unitac_ecf_adapter import UniTac_ECFAdapter
    ecf = UniTac_ECFAdapter()
    assert hasattr(ecf, "load"), "Missing load()"
    assert hasattr(ecf, "encode"), "Missing encode()"
    assert hasattr(ecf, "_backend"), "Missing _backend"
    print("  ✅ UniTac_ECFAdapter: load() / encode() interface complete")

    # 15. UniTacECFRunner interface
    from benchlink.benchmarks.unitac_ecf_runner import UniTacECFRunner
    uer = UniTacECFRunner()
    assert hasattr(uer, "setup"), "Missing setup()"
    assert hasattr(uer, "evaluate"), "Missing evaluate()"
    print("  ✅ UniTacECFRunner: setup() / evaluate() interface complete")

    # 16. OpenPiAdapter interface
    from benchlink.models.openpi_adapter import OpenPiAdapter
    op = OpenPiAdapter()
    assert hasattr(op, "load"), "Missing load()"
    assert hasattr(op, "act"), "Missing act()"
    assert hasattr(op, "reset"), "Missing reset()"
    print("  ✅ OpenPiAdapter: load() / act() / reset() interface complete")

    # 17. MotusAdapter interface
    from benchlink.models.motus_adapter import MotusAdapter
    mo = MotusAdapter()
    assert hasattr(mo, "load"), "Missing load()"
    assert hasattr(mo, "act"), "Missing act()"
    assert hasattr(mo, "reset"), "Missing reset()"
    print("  ✅ MotusAdapter: load() / act() / reset() interface complete")

    # 18. DreamZeroAdapter interface
    from benchlink.models.dreamzero_adapter import DreamZeroAdapter
    dz = DreamZeroAdapter()
    assert hasattr(dz, "load"), "Missing load()"
    assert hasattr(dz, "act"), "Missing act()"
    assert hasattr(dz, "reset"), "Missing reset()"
    print("  ✅ DreamZeroAdapter: load() / act() / reset() interface complete")


# ── Action line verification: MockFastWAM + MockLiberoEnv ──
def test_action_line():
    print("\n" + "=" * 60)
    print("[3/4] Action Line Verification: FastWAMAdapter × LiberoRunner")
    print("=" * 60)

    from benchlink.base import ModelAdapter
    from benchlink.schema import CanonicalObs

    # ── Mock FastWAM model ──
    class MockFastWAM(ModelAdapter):
        """Mock FastWAM: generates random delta actions from proprio."""

        def load(self, checkpoint: str, config: dict) -> None:
            self.device = config.get("device", "cpu")

        def act(self, obs: CanonicalObs) -> np.ndarray:
            # Output a reasonable standard action
            return np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        def reset(self) -> None:
            pass

    # ── Mock LIBERO environment ──
    class MockLiberoEnv:
        """Mock LIBERO OffScreenRenderEnv (with _env wrapper layer)."""

        def __init__(self, task_description="test_task", n_steps=5):
            self.task_description = task_description
            self._step_count = 0
            self.n_steps = n_steps
            jp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._joint_pos = jp.copy()
            # LIBERO OffScreenRenderEnv accesses the underlying environment via self._env
            self._env = MockLiberoInnerEnv(self)

        def reset(self):
            self._step_count = 0
            self._joint_pos[:] = 0.0
            return self._make_obs(), {}

        def step(self, action):
            self._step_count += 1
            self._joint_pos += action[:7]  # Simple integration
            done = self._step_count >= self.n_steps
            return self._make_obs(), 0.0, done, False, {"success": done}

        def seed(self, seed):
            pass

        def get_task_embedding(self):
            return np.zeros(512, dtype=np.float32)

        def close(self):
            pass

        def _make_obs(self):
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_joint_pos": self._joint_pos.copy(),
            }

    class MockLiberoInnerEnv:
        """Mock LIBERO internal _env object (for joint_pos access)."""
        def __init__(self, outer):
            self._outer = outer
        def get_robot0_joint_positions(self):
            return self._outer._joint_pos.copy()

    # ── Assemble LiberoRunner + Mock environment ──
    from benchlink.benchmarks.libero_runner import LiberoRunner

    runner = LiberoRunner()
    runner.env = MockLiberoEnv(task_description="test_task", n_steps=5)
    runner.task_name = "test_task"
    runner.task_description = "test_task"
    runner._max_steps = 10
    runner.config = {}

    # ── Wire up MockFastWAM ──
    model = MockFastWAM()
    model.load("", {"device": "cpu"})

    # ── Run minimal evaluation ──
    print("  Running minimal eval with 3 episodes...")
    result = runner.evaluate(model, n_episodes=3)

    print(f"  Result: {json.dumps(result, indent=4)}")

    assert "success_rate" in result, "Missing success_rate"
    assert result["n_episodes"] == 3, f"Expected 3 episodes, got {result['n_episodes']}"
    assert result["total_steps"] > 0, "total_steps should be positive"
    print("  ✅ Action line verification passed: interface data flow complete")


# ── Tactile line verification: MockAnyTouch + MockProbe ──
def test_tactile_line():
    print("\n" + "=" * 60)
    print("[4/4] Tactile Line Verification: AnyTouchAdapter × AnyTouchProbeRunner")
    print("=" * 60)

    from benchlink.base import TactileAdapter

    # ── Mock AnyTouch encoder ──
    class MockAnyTouch(TactileAdapter):
        """Mock AnyTouch: returns fixed-dimension random feature vectors."""

        def load(self, checkpoint: str, config: dict) -> None:
            self.feat_dim = config.get("feat_dim", 768)
            self.device = config.get("device", "cpu")

        def encode(self, tactile_img: np.ndarray) -> np.ndarray:
            # Simulate feature output; first 20 dims encode image mean (so similar images yield similar features)
            feat = np.random.randn(self.feat_dim).astype(np.float32)
            feat[:20] = tactile_img.mean()  # Make input affect output
            return feat

    # ── Construct minimal probe dataset ──
    from benchlink.benchmarks.anytouch_probe_runner import AnyTouchProbeRunner

    probe = AnyTouchProbeRunner()

    # Inject synthetic dataset: 3 classes, 10 images each (uint8 max 255)
    probe._datasets["test_tag"] = [
        (np.full((64, 64, 3), fill_value=min(i * 25, 255), dtype=np.uint8), i // 10)
        for i in range(30)
    ]
    probe._datasets["test_of1"] = [
        (np.full((64, 64, 3), fill_value=min(50 + i, 255), dtype=np.uint8), i % 3)
        for i in range(30)
    ]

    # ── Wire up MockAnyTouch ──
    model = MockAnyTouch()
    model.load("", {"feat_dim": 64, "device": "cpu"})  # Use 64 dims for speed

    # Override feat_dim so encode returns 64 dims
    model.feat_dim = 64

    # ── Run minimal evaluation ──
    print("  Running probe eval (2 subsets, 30 samples/subset)...")
    result = probe.evaluate(model)

    print(f"  Result: {json.dumps(result, indent=4)}")

    assert "test_tag" in result, "Missing test_tag"
    assert "test_of1" in result, "Missing test_of1"
    assert "mean" in result, "Missing mean"
    print(f"  Mean accuracy: {result['mean']:.4f}")
    print("  ✅ Tactile line verification passed: interface data flow complete")


# ── CLI entry point test ──
def test_cli_entry():
    print("\n" + "=" * 60)
    print("[Extra] CLI --list test")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "benchlink.cli", "--list"],
        capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    print(f"  stdout:\n{result.stdout}")
    print("  ✅ CLI --list OK")


# ── Main entry point ──
def main():
    print("=" * 60)
    print("BenchLink Eval Framework — Phase 1 Minimal Verification")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD:    {Path.cwd()}")
    print("=" * 60)

    # Ensure benchlink package root is on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    tests = [
        ("Registration Test", test_registration),
        ("Interface Signature Test", test_interface_signatures),
        ("Action Line Verification", test_action_line),
        ("Tactile Line Verification", test_tactile_line),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ {name} failed: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
