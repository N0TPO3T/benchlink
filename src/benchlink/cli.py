#!/usr/bin/env python3
"""
CLI entry point: unified evaluation for any baseline x any benchmark.

Usage:
    # List registered components
    benchlink --list
    python -m benchlink.cli --list

    # Run evaluation
    benchlink \\
        --model rdp \\
        --benchmark manifeel \\
        --checkpoint /path/to/model.ckpt \\
        --config configs/default.yaml \\
        --episodes 5

    # Override data path or device
    benchlink \\
        --model rdp \\
        --benchmark manifeel \\
        --checkpoint /path/to/model.ckpt \\
        --config configs/default.yaml \\
        --data-root /path/to/data \\
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from benchlink.registry import (
    get_adapter,
    get_tactile,
    get_runner,
    list_models,
    list_tactile_models,
    list_benchmarks,
)
from benchlink import registry as _registry


def load_config(path: str) -> dict:
    """Load a yaml or json config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = p.read_text()
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("PyYAML required for .yaml config: pip install pyyaml")
        return yaml.safe_load(content)
    elif path.endswith(".json"):
        return json.loads(content)
    else:
        try:
            if yaml:
                return yaml.safe_load(content)
        except Exception:
            pass
        return json.loads(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BenchLink — Unified Baseline x Benchmark Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Main arguments
    parser.add_argument("--model", type=str, help="Model name (e.g. rdp, fastwam)")
    parser.add_argument("--benchmark", type=str, help="Benchmark name (e.g. manifeel, libero)")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")

    # Config overrides
    parser.add_argument("--data-root", type=str, default=None, help="Override data path")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")

    # Info
    parser.add_argument("--list", action="store_true", help="List registered models & benchmarks")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── --list mode ──
    if args.list:
        models = list_models()
        tactiles = list_tactile_models()
        benches = list_benchmarks()
        print(f"Registered models ({len(models)}):")
        for name in sorted(models):
            print(f"  - {name}")
        print(f"Registered tactile models ({len(tactiles)}):")
        for name in sorted(tactiles):
            print(f"  - {name}")
        print(f"Registered benchmarks ({len(benches)}):")
        for name in sorted(benches):
            print(f"  - {name}")
        print()
        print("Example commands:")
        print("  benchlink --model fastwam --benchmark libero --checkpoint <path> --config <file>")
        print("  benchlink --model anytouch --benchmark anytouch_probe --checkpoint <path> --config <file>")
        print("  benchlink --model vla_touch --benchmark maniskill --checkpoint <path> --config <file>")
        return

    # ── Validate arguments ──
    if not args.model:
        parser.error("--model is required (use --list to see available)")
    if not args.benchmark:
        parser.error("--benchmark is required (use --list to see available)")
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if not args.config:
        default_cfg = Path(__file__).parent / "configs" / "default.yaml"
        if default_cfg.exists():
            args.config = str(default_cfg)
        else:
            parser.error("--config is required")

    # ── Load config ──
    cfg = load_config(args.config)

    # ── CLI overrides ──
    if args.data_root:
        cfg["data_path"] = args.data_root
    if args.device:
        cfg["device"] = args.device

    # ── Initialize Runner ──
    runner_cls = get_runner(args.benchmark)
    runner = runner_cls()
    runner.setup(cfg)
    print(f"[Eval] Initialized benchmark '{args.benchmark}'", file=sys.stderr)

    # ── Initialize Adapter (auto-detect action vs tactile) ──
    # Check ModelAdapter registry first, fall back to TactileAdapter
    if args.model in _registry._MODEL_REGISTRY:
        adapter_cls = get_adapter(args.model)
        model = adapter_cls()
        model.load(args.checkpoint, cfg)
        model_type = "model"
    elif args.model in _registry._TACTILE_REGISTRY:
        adapter_cls = get_tactile(args.model)
        model = adapter_cls()
        model.load(args.checkpoint, cfg)
        model_type = "tactile model"
    else:
        available = list(_registry._MODEL_REGISTRY.keys()) + list(_registry._TACTILE_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{args.model}'. "
            f"Available (action + tactile): {available}"
        )
    print(f"[Eval] Initialized {model_type} '{args.model}'", file=sys.stderr)

    # ── Run evaluation ──
    print(f"[Eval] Running {args.episodes} episodes...", file=sys.stderr)
    result = runner.evaluate(model, n_episodes=args.episodes, device=cfg.get("device", "cuda"))

    # ── Output results ──
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
