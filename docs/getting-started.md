# Getting Started

## Installation

```bash
pip install benchlink
```

For additional features:

```bash
# Tactile encoder support (torch, timm, transformers, scikit-learn)
pip install "benchlink[tactile]"

# All optional dependencies
pip install "benchlink[all]"
```

## Quick Start

### 1. List Available Components

```bash
benchlink --list
```

### 2. Run the Zero-Dependency Demo

```bash
git clone https://github.com/N0TPO3T/benchlink
cd benchlink
PYTHONPATH=. python examples/dummy_adapter.py
PYTHONPATH=. python examples/dummy_runner.py
```

### 3. Run a Real Evaluation

You need:
- A model checkpoint
- Benchmark data
- A configuration file

```bash
benchlink --model fastwam --benchmark libero \
  --checkpoint /path/to/fastwam.ckpt \
  --config configs/libero.yaml \
  --episodes 10
```

### 4. Using Docker-based Models

For models like OpenPI, Motus, or DreamZero that run in Docker containers:

```bash
# 1. Pull the Docker image
docker pull <image>

# 2. Start the inference server container (see docs/docker-deployment.md)

# 3. Run evaluation
benchlink --model dreamzero --benchmark maniskill \
  --checkpoint "" \
  --config configs/default.yaml --episodes 5
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **CanonicalObs** | Standard observation with 11 fields (vision, proprioception, tactile, language) |
| **ModelAdapter** | Wraps a model's inference (load → act → reset) |
| **BenchmarkRunner** | Wraps a benchmark's evaluation (setup → evaluate → canonical conversion) |
| **Registry** | Central catalog of all registered adapters and runners |
| **Standard Action** | (7,) numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper] |

## Next Steps

- Browse [adapters.md](adapters.md) for all available model adapters
- Browse [benchmarks.md](benchmarks.md) for all available benchmark runners
- Learn how to [add a new adapter](contributing.md#adding-a-new-model-adapter)
- Set up [Docker deployment](docker-deployment.md) for containerized models
