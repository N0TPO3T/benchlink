# BenchLink

**Unified evaluation framework decoupling robot models from benchmarks through a canonical observation contract.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

BenchLink provides a standardized interface for evaluating **any robot model** (action policies, tactile encoders) against **any benchmark** (simulation environments, offline datasets, linear probes). The core idea: define a fixed `CanonicalObs` data contract, wrap each model in an **Adapter**, wrap each benchmark in a **Runner**, and let them communicate through the registry without knowing each other's internals.

## Quick Start

```bash
pip install benchlink

# See available components
benchlink --list
```

```
Registered models (8):
  - dp, dreamzero, fastwam, motus, openpi, rdp, rdt, vla_touch
Registered tactile models (5):
  - anytouch, sparsh, t3, unitac_ecf, vla_touch
Registered benchmarks (8):
  - anytouch_probe, droid_sim, libero, manifeel, manifeel_sim, maniskill, robotwin, unitac_ecf
```

### Zero-dependency Demo

```python
from benchlink.base import ModelAdapter, BenchmarkRunner
from benchlink.schema import CanonicalObs

class DummyAdapter(ModelAdapter):
    def load(self, checkpoint, config):
        self._action = [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5]
    def act(self, obs): return self._action
    def reset(self): pass

# Run examples:
PYTHONPATH=. python examples/dummy_adapter.py
PYTHONPATH=. python examples/dummy_runner.py
```

### Real Evaluation

```bash
# Action model -> simulation benchmark
benchlink --model fastwam --benchmark libero \
  --checkpoint /path/to/weights.ckpt \
  --config configs/libero.yaml --episodes 10

# Tactile encoder -> linear probe
benchlink --model anytouch --benchmark anytouch_probe \
  --checkpoint /path/to/encoder.pt \
  --config configs/anytouch.yaml
```

## Architecture

```
CLI (cli.py)
  |
  +-- Registry (registry.py)
  |     +-- _MODEL_REGISTRY     -> ModelAdapter (act)
  |     +-- _TACTILE_REGISTRY   -> TactileAdapter (encode)
  |     +-- _BENCHMARK_REGISTRY -> BenchmarkRunner (evaluate)
  |
  +-- CanonicalObs (schema.py)
  |     +-- Standard observation data contract
  |
  +-- Base Classes (base.py)
        +-- ModelAdapter     -- action model (load / act / reset)
        +-- TactileAdapter   -- tactile encoder (load / encode)
        +-- BenchmarkRunner  -- evaluation executor (setup / evaluate)
```

### Data Flow

```
+---------------+   CanonicalObs    +---------------+
| Benchmark     | ----------------> | ModelAdapter  |
| Runner        |   rgb_static,     |   .act()      |
|               |   proprio,        |               |
|  .evaluate()  |   language, ...   |   -> action   |
|               | <---------------- |   (7,)        |
+---------------+      action       +---------------+
```

### CanonicalObs Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `rgb_static` | (H, W, 3) | Fixed-view RGB |
| `rgb_gripper` | (H, W, 3) | Wrist camera RGB |
| `depth` | (H, W) | Depth map |
| `proprio` | (N,) | Joint positions or end-effector pose |
| `tactile_img` | (H, W, 3) | Tactile image |
| `tactile_feat` | (64,) | Pre-extracted tactile features |
| `tactile_depth` | (H, W) | Tactile depth |
| `tactile_force` | (H, W, 3) | Tactile force field |
| `language` | str | Task description |
| `lang_embed` | (512,) | Pre-computed instruction embedding |
| `extra` | dict | Extension fields |

Standard action: `(7,)` → `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

## Registered Components

### Action Models (8)

| Name | Class | Loading | Notes |
|------|-------|---------|-------|
| `rdp` | RDPAdapter | Subprocess (conda) | Reactive Diffusion Policy |
| `fastwam` | FastWAMAdapter | Direct import | Fast World Action Model |
| `dp` | DPAdapter | Direct import | Diffusion Policy |
| `openpi` | OpenPiAdapter | Docker exec | pi0 VLA |
| `motus` | MotusAdapter | Docker exec | Mixture-of-Transformers WAM |
| `dreamzero` | DreamZeroAdapter | Docker exec | NVIDIA GEAR WAM |
| `rdt` | RDTAdapter | Direct import | Diffusion Transformer 1B |
| `vla_touch` | VLA_TouchAdapter | Direct import | VLA + tactile (also registered as TactileAdapter) |

### Tactile Encoders (5)

| Name | Class | Dim | Notes |
|------|-------|-----|-------|
| `anytouch` | AnyTouchAdapter | 768 | CLIP ViT-L/14 (ICLR 2025) |
| `sparsh` | SparshAdapter | 768 | DINOv2 backbone |
| `t3` | T3Adapter | 1024 | ResNet/CNN/ViT/MAE |
| `vla_touch` | VLA_TouchAdapter | 768 | Dual-registered |
| `unitac_ecf` | UniTac_ECFAdapter | configurable | Runtime delegation |

### Benchmarks (8)

| Name | Class | Type | Metrics |
|------|-------|------|---------|
| `libero` | LiberoRunner | Simulation | success_rate |
| `maniskill` | ManiSkillRunner | Simulation | success_rate |
| `droid_sim` | DroidSimRunner | Simulation | success_rate |
| `manifeel_sim` | ManiFeelSimRunner | Simulation (IsaacGym) | success_rate |
| `manifeel` | ManiFeelRunner | Offline replay | position/rotation error |
| `robotwin` | RoboTwinRunner | Offline replay | position/rotation error |
| `anytouch_probe` | AnyTouchProbeRunner | Linear probe | accuracy per subset |
| `unitac_ecf` | UniTacECFRunner | Linear probe | accuracy per dataset |

## Extending

### Add a New Model

```python
from benchlink.base import ModelAdapter
from benchlink.registry import register_model

@register_model("my_model", ...)  # or register_model("my_model", MyAdapter)
class MyAdapter(ModelAdapter):
    def load(self, checkpoint, config): ...
    def act(self, obs): ...
    def reset(self): ...
```

### Add a New Benchmark

```python
from benchlink.base import BenchmarkRunner
from benchlink.registry import register_benchmark

@register_benchmark("my_bench", ...)
class MyRunner(BenchmarkRunner):
    def setup(self, config): ...
    def evaluate(self, model, n_episodes=10, **kwargs): ...
    def _to_canonical(self, raw_obs): ...
```

## Docker Deployment

For models that run in isolated environments (OpenPI, Motus, DreamZero), BenchLink provides:

1. **DockerModelAdapter** base class — handles container lifecycle, JSON Lines protocol
2. **Inference servers** — standalone scripts inside containers, communicate via stdin/stdout
3. **Server-side deploy script** — `python -m benchlink.deploy.deploy_inference_servers.sh`

See [Docker Deployment Guide](docs/docker-deployment.md) for details.

## Documentation

- [Architecture](docs/architecture.md) — Full design overview
- [Getting Started](docs/getting-started.md) — 5-minute guide
- [Adapter Catalog](docs/adapters.md) — All model adapters in detail
- [Benchmark Catalog](docs/benchmarks.md) — All benchmark runners in detail
- [Server Protocol](docs/servers.md) — JSON Lines inference protocol spec
- [Contributing](docs/contributing.md) — How to extend BenchLink
- 中文设计文档: [docs/zh/](docs/zh/) (Chinese design documents)

## License

Apache 2.0 — see [LICENSE](LICENSE).

Third-party model weights referenced by adapters are subject to their original licenses; see [Third-Party Notices](docs/third-party-notices.md).
