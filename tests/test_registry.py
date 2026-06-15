"""Test registry: all components register correctly."""

from benchlink.registry import (
    list_models, list_tactile_models, list_benchmarks,
    get_adapter, get_tactile, get_runner,
)


class TestRegistration:
    """Verify all components are registered."""

    def test_action_models_registered(self):
        models = list_models()
        required = ["fastwam", "dp", "rdp", "openpi", "motus", "dreamzero", "vla_touch", "rdt"]
        for name in required:
            assert name in models, f"{name} not registered"

    def test_tactile_models_registered(self):
        tactiles = list_tactile_models()
        required = ["anytouch", "sparsh", "t3", "vla_touch", "unitac_ecf"]
        for name in required:
            assert name in tactiles, f"{name} not registered"

    def test_benchmarks_registered(self):
        benches = list_benchmarks()
        required = ["libero", "anytouch_probe", "maniskill", "robotwin",
                     "droid_sim", "unitac_ecf", "manifeel", "manifeel_sim"]
        for name in required:
            assert name in benches, f"{name} not registered"

    def test_get_adapter(self):
        cls = get_adapter("fastwam")
        assert cls.__name__ == "FastWAMAdapter"

    def test_get_runner(self):
        cls = get_runner("libero")
        assert cls.__name__ == "LiberoRunner"
