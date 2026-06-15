# Benchmark Catalog

## Simulation Benchmarks

### LIBERO
- **Class**: `LiberoRunner`
- **File**: `benchmarks/libero_runner.py`
- **Type**: Online simulation (OffScreenRenderEnv)
- **Data**: ~30GB (LIBERO-90 task suite)
- **Metrics**: `success_rate`
- **Usage**: `benchlink --model <action> --benchmark libero`

### ManiSkill
- **Class**: `ManiSkillRunner`
- **File**: `benchmarks/maniskill_runner.py`
- **Type**: Online simulation (SAPIEN + MuJoCo)
- **Tasks**: PickCube, StackCube, PegInsertionSide
- **Metrics**: `success_rate`

### Droid Sim
- **Class**: `DroidSimRunner`
- **File**: `benchmarks/droid_sim_runner.py`
- **Type**: Online simulation
- **Backend**: `sim_evals` or `droid_sim` package
- **Metrics**: `success_rate`

### ManiFeel Sim
- **Class**: `ManiFeelSimRunner`
- **File**: `benchmarks/manifeel_sim_runner.py`
- **Type**: Online simulation (IsaacGym via Docker)
- **Requires**: `manifeel:l2-code` Docker image (private)
- **Protocol**: JSON Lines with `__SIM__` prefix
- **Metrics**: `success_rate`

## Offline Benchmarks

### ManiFeel
- **Class**: `ManiFeelRunner`
- **File**: `benchmarks/manifeel_runner.py`
- **Type**: Offline replay from zarr
- **Data**: ~24GB (USB insertion task)
- **Metrics**: `position_error_mean`, `rotation_error_mean`

### RoboTwin
- **Class**: `RoboTwinRunner`
- **File**: `benchmarks/robotwin_runner.py`
- **Type**: Offline replay from zarr
- **Data**: ~75GB
- **Features**: Multi-key fallback for different data versions
- **Metrics**: `position_error_mean`, `rotation_error_mean`
- **Notes**: Has `close()` method for cleanup

## Probe Benchmarks

### AnyTouch Probe
- **Class**: `AnyTouchProbeRunner`
- **File**: `benchmarks/anytouch_probe_runner.py`
- **Type**: Linear probe on tactile datasets
- **Datasets**: TAG, OF1, OF2, Feel
- **Metrics**: Accuracy per subset

### UniTac_ECF Probe
- **Class**: `UniTacECFRunner`
- **File**: `benchmarks/unitac_ecf_runner.py`
- **Type**: Linear probe on multi-dataset
- **Datasets**: TAG, OF1, OF2, Feel, VitacLab, FOTA
- **Metrics**: Accuracy per dataset
