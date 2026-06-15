"""
Registry — manages registration and lookup of all ModelAdapters, TactileAdapters, and BenchmarkRunners.
"""

from typing import Dict, Type

from .base import ModelAdapter, TactileAdapter, BenchmarkRunner


# ── Global registries ──
_MODEL_REGISTRY: Dict[str, Type[ModelAdapter]] = {}
_TACTILE_REGISTRY: Dict[str, Type[TactileAdapter]] = {}
_BENCHMARK_REGISTRY: Dict[str, Type[BenchmarkRunner]] = {}


# ── Registration ──

def register_model(name: str, cls: Type[ModelAdapter]) -> None:
    """Register an action model adapter."""
    if name in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' already registered")
    _MODEL_REGISTRY[name] = cls


def register_tactile(name: str, cls: Type[TactileAdapter]) -> None:
    """Register a tactile encoder adapter."""
    if name in _TACTILE_REGISTRY:
        raise KeyError(f"Tactile model '{name}' already registered")
    _TACTILE_REGISTRY[name] = cls


def register_benchmark(name: str, cls: Type[BenchmarkRunner]) -> None:
    """Register a benchmark runner."""
    if name in _BENCHMARK_REGISTRY:
        raise KeyError(f"Benchmark '{name}' already registered")
    _BENCHMARK_REGISTRY[name] = cls


# ── Lookup ──

def get_adapter(model_name: str) -> Type[ModelAdapter]:
    """Look up a ModelAdapter class by name."""
    cls = _MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return cls


def get_tactile(model_name: str) -> Type[TactileAdapter]:
    """Look up a TactileAdapter class by name."""
    cls = _TACTILE_REGISTRY.get(model_name)
    if cls is None:
        raise KeyError(
            f"Unknown tactile model '{model_name}'. "
            f"Available: {list(_TACTILE_REGISTRY.keys())}"
        )
    return cls


def get_runner(benchmark_name: str) -> Type[BenchmarkRunner]:
    """Look up a BenchmarkRunner class by name."""
    cls = _BENCHMARK_REGISTRY.get(benchmark_name)
    if cls is None:
        raise KeyError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available: {list(_BENCHMARK_REGISTRY.keys())}"
        )
    return cls


# ── Listing ──

def list_models() -> list:
    return list(_MODEL_REGISTRY.keys())


def list_tactile_models() -> list:
    return list(_TACTILE_REGISTRY.keys())


def list_benchmarks() -> list:
    return list(_BENCHMARK_REGISTRY.keys())
