"""Test ABC interface signatures for all adapters and runners."""

import pytest


# Each (module_path, class_name, required_methods)
MODEL_ADAPTERS = [
    ("benchlink.models.fastwam_adapter", "FastWAMAdapter", ["load", "act", "reset"]),
    ("benchlink.models.dp_adapter", "DPAdapter", ["load", "act", "reset"]),
    ("benchlink.models.rdp_adapter", "RDPAdapter", ["load", "act", "reset"]),
    ("benchlink.models.openpi_adapter", "OpenPiAdapter", ["load", "act", "reset"]),
    ("benchlink.models.motus_adapter", "MotusAdapter", ["load", "act", "reset"]),
    ("benchlink.models.dreamzero_adapter", "DreamZeroAdapter", ["load", "act", "reset"]),
    ("benchlink.models.rdt_adapter", "RDTAdapter", ["load", "act", "reset"]),
    ("benchlink.models.vla_touch_adapter", "VLA_TouchAdapter", ["load", "act", "reset"]),
]

TACTILE_ADAPTERS = [
    ("benchlink.models.anytouch_adapter", "AnyTouchAdapter", ["load", "encode"]),
    ("benchlink.models.sparsh_adapter", "SparshAdapter", ["load", "encode"]),
    ("benchlink.models.t3_adapter", "T3Adapter", ["load", "encode"]),
    ("benchlink.models.unitac_ecf_adapter", "UniTac_ECFAdapter", ["load", "encode"]),
]

BENCHMARK_RUNNERS = [
    ("benchlink.benchmarks.libero_runner", "LiberoRunner",
     ["setup", "evaluate", "_to_canonical", "_from_canonical"]),
    ("benchlink.benchmarks.maniskill_runner", "ManiSkillRunner",
     ["setup", "evaluate", "_to_canonical", "_from_canonical"]),
    ("benchlink.benchmarks.manifeel_runner", "ManiFeelRunner",
     ["setup", "evaluate", "_to_canonical", "_from_canonical"]),
    ("benchlink.benchmarks.robotwin_runner", "RoboTwinRunner",
     ["setup", "evaluate", "_to_canonical", "_from_canonical", "close"]),
    ("benchlink.benchmarks.droid_sim_runner", "DroidSimRunner",
     ["setup", "evaluate", "_to_canonical", "_from_canonical"]),
    ("benchlink.benchmarks.anytouch_probe_runner", "AnyTouchProbeRunner",
     ["setup", "evaluate"]),
]


@pytest.mark.parametrize("module_path,cls_name,methods", MODEL_ADAPTERS)
def test_model_adapter_interface(module_path, cls_name, methods):
    """Verify each ModelAdapter has the required methods."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    instance = cls()
    for method in methods:
        assert hasattr(instance, method), f"{cls_name} missing {method}()"


@pytest.mark.parametrize("module_path,cls_name,methods", TACTILE_ADAPTERS)
def test_tactile_adapter_interface(module_path, cls_name, methods):
    """Verify each TactileAdapter has the required methods."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    instance = cls()
    for method in methods:
        assert hasattr(instance, method), f"{cls_name} missing {method}()"


@pytest.mark.parametrize("module_path,cls_name,methods", BENCHMARK_RUNNERS)
def test_benchmark_runner_interface(module_path, cls_name, methods):
    """Verify each BenchmarkRunner has the required methods."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    instance = cls()
    for method in methods:
        assert hasattr(instance, method), f"{cls_name} missing {method}()"
