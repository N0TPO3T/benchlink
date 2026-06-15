# Changelog

## 0.1.0.dev0 (2026-06-15)

- Initial release candidate
- Core framework: ModelAdapter, TactileAdapter, BenchmarkRunner ABCs
- CanonicalObs data contract with 11 observation fields
- Registry pattern for extensible component management
- CLI entry point with automatic action/tactile model detection
- 8 action model adapters (RDP, FastWAM, DP, OpenPI, Motus, DreamZero, RDT, VLA-Touch)
- 5 tactile encoder adapters (AnyTouch, Sparsh, T3, VLA-Touch, UniTac_ECF)
- 8 benchmark runners (ManiFeel, LIBERO, ManiSkill, RoboTwin, DroidSim, AnyTouchProbe, UniTac_ECF, ManiFeelSim)
- DockerModelAdapter base class with JSON Lines inference protocol
- 4 standalone inference servers (RDP, OpenPI, Motus, DreamZero)
- Zero-dependency DummyAdapter/DummyRunner examples
- Interface verification test suite (no GPU/Docker/network required)
- Chinese design documentation (docs/zh/)
