# Architecture

## Overview

BenchLink decouples robot models from evaluation benchmarks through a three-layer architecture:

```
           CanonicalObs (schema.py)
                ↕
    BenchmarkRunner  ←→  ModelAdapter
```

The layers are:

1. **CanonicalObs** — A fixed-format data contract (11 optional fields covering vision, proprioception, tactile, and language)
2. **BenchmarkRunner** — One per benchmark, wraps the simulation environment or offline dataset. Converts native observations to/from CanonicalObs.
3. **ModelAdapter** — One per baseline, wraps model inference. Receives CanonicalObs, produces standard actions (or tactile features).

## Registry Pattern

Three global dictionaries hold registered component classes:

- `_MODEL_REGISTRY` — action model adapters (8 registered)
- `_TACTILE_REGISTRY` — tactile encoder adapters (5 registered)
- `_BENCHMARK_REGISTRY` — benchmark runners (8 registered)

Registration happens at import time via `__init__.py` files. The CLI auto-detects whether a model is an action model or tactile model by checking `_MODEL_REGISTRY` first, then `_TACTILE_REGISTRY`.

## Standard Action Space

```
action: np.ndarray, shape=(7,)
[dx, dy, dz, droll, dpitch, dyaw, gripper]
- First 6 dims: end-effector delta pose
- 7th dim: gripper in [0, 1]
```

This 7-D delta pose was validated as a universal interchange format across 7 different action models (RDP TCP absolute → 7-D delta converter, FastWAM joint position → delta pose, etc.).

## Docker Adapter Architecture

For models that cannot be directly imported (heavy dependencies, hardware requirements), BenchLink provides a Docker-based adapter pattern:

```
┌─────────────────────────────┐     stdin/stdout      ┌──────────────────────┐
│   ModelAdapter (Python)     │ ◄──── JSON Lines ──── │  Docker Container    │
│                             │                        │  ┌────────────────┐  │
│  _send_request(obs)         │ ──→ {"obs": ...,       │  │ Inference      │  │
│                             │       "id": N}         │  │ Server (Python)│  │
│  ←── {"action": [...],      │ ←──                    │  │                │  │
│         "id": N}            │                        │  │ load model     │  │
└─────────────────────────────┘                        │  │ act(obs)       │  │
                                                       └──┴────────────────┘──┘
```

The protocol is a simple JSON Lines exchange:
- Ready signal: `{"status": "ready"}`
- Inference request: `{"obs": {...}, "id": N}`
- Inference response: `{"action": [...], "id": N}`
- Reset signal: `{"reset": true, "id": N}`

## Design Decisions

1. **Why CanonicalObs instead of a list dict?** — Type safety, IDE autocompletion, explicit field documentation.

2. **Why delta pose as standard action?** — Most manipulation benchmarks and models can express actions as end-effector delta poses. Joint-level policies (FastWAM) need a joint-to-delta converter, but the converter is simple and the benefit of a universal action space outweighs the cost.

3. **Why registry instead of plugin discovery?** — Explicit registration in `__init__.py` makes the component inventory transparent and avoids import-time side effects from filesystem scanning.

4. **Why both ModelAdapter and TactileAdapter?** — Action models and tactile encoders have fundamentally different interfaces (act vs encode). Forcing both into one ABC would leak abstraction. The CLI auto-detects which one to use.

## Verification

The architecture was validated end-to-end with RDP + ManiFeel (both offline replay and online IsaacGym simulation). Key findings:

- All 6 bugs encountered during integration were in adapter/runner implementation details, **none** in the interface design
- Interface signatures never changed from first commit to final test
- CanonicalObs handles missing fields gracefully (None values cause no errors in downstream code)
- 7-D delta pose works as a cross-model interchange format
- Docker JSON Lines protocol is stable with 3GB model loading
