# Contributing to BenchLink

## Adding a New Model Adapter

1. Create a new file in `src/benchlink/models/<name>_adapter.py`
2. Implement the `ModelAdapter` or `TactileAdapter` ABC:

```python
from benchlink.base import ModelAdapter
from benchlink.schema import CanonicalObs

class MyAdapter(ModelAdapter):
    def load(self, checkpoint: str, config: dict) -> None:
        # Load weights, set up model
        pass

    def act(self, obs: CanonicalObs) -> np.ndarray:
        # Run inference, return (7,) action
        pass

    def reset(self) -> None:
        # Reset episode state
        pass
```

3. Register in `src/benchlink/models/__init__.py`:
```python
from benchlink.models.my_adapter import MyAdapter
register_model("my_model", MyAdapter)
```

4. Verify registration:
```bash
benchlink --list  # Should show "my_model"
```

## Adding a New Benchmark Runner

1. Create `src/benchlink/benchmarks/<name>_runner.py`
2. Implement `BenchmarkRunner`:

```python
from benchlink.base import BenchmarkRunner
from benchlink.schema import CanonicalObs

class MyRunner(BenchmarkRunner):
    def setup(self, config: dict) -> None:
        # Load data or create environment
        pass

    def evaluate(self, model, n_episodes=10, **kwargs):
        # Run evaluation loop
        pass

    def _to_canonical(self, raw_obs) -> CanonicalObs:
        # Convert native observation
        pass
```

3. Register in `src/benchlink/benchmarks/__init__.py`:
```python
register_benchmark("my_bench", MyRunner)
```

## Code Style

- Python 3.10+ type annotations
- Use `ruff` for linting
- Keep docstrings in English
- No hardcoded paths — use config keys instead

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Verify interface contracts
python tests/verify_phase1.py
```

## Pull Request Process

1. Ensure tests pass (`pytest tests/ -v`)
2. Run `ruff check src/`
3. Update `CHANGELOG.md` if adding significant features
4. Open PR with clear description of changes
