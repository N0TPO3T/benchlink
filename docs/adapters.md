# Adapter Catalog

## Action Models (ModelAdapter)

### RDP (Reactive Diffusion Policy)
- **Class**: `RDPAdapter`
- **File**: `models/rdp_adapter.py`
- **Loading**: Subprocess (conda environment)
- **Weights**: ~3GB (LDP + AT checkpoints)
- **Protocol**: Stdio JSON Lines via persistent subprocess
- **Action conversion**: TCP absolute position → 7-D delta pose
- **Reference**: [reactive_diffusion_policy](https://github.com/WendiChen/reactive_diffusion_policy_model)

### FastWAM
- **Class**: `FastWAMAdapter`
- **File**: `models/fastwam_adapter.py`
- **Loading**: Direct import
- **Weights**: Needs `.ckpt` file
- **Action conversion**: Joint position → delta pose
- **Notes**: No gripper rotation support (zeroed)

### DP (Diffusion Policy)
- **Class**: `DPAdapter`
- **File**: `models/dp_adapter.py`
- **Loading**: Direct import (`diffusion_policy` package)
- **Features**: Obs history buffer, action chunk caching

### OpenPI (pi0)
- **Class**: `OpenPiAdapter`
- **File**: `models/openpi_adapter.py`
- **Loading**: Docker exec (`openpi_server` container)
- **Protocol**: JSON Lines via `docker exec`
- **Weights**: pi0 base checkpoint (~12GB)
- **Reference**: [physical-intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

### Motus
- **Class**: `MotusAdapter`
- **File**: `models/motus_adapter.py`
- **Loading**: Docker exec (`motus_server` container)
- **Weights**: Wan2.2-TI2V-5B (~10GB) + T5 encoder
- **Features**: Image sequence buffer, action chunk caching

### DreamZero
- **Class**: `DreamZeroAdapter`
- **File**: `models/dreamzero_adapter.py`
- **Loading**: Docker exec (`dreamzero_server` container)
- **Weights**: 16.5B model (INT8 quantized, ~15.7GB)
- **Features**: VAE latent decoding, latent history buffer
- **Reference**: [NVIDIA/GEAR Lab DreamZero](https://research.nvidia.com/labs/gear/dreamzero/)

### RDT (Robotics Diffusion Transformer)
- **Class**: `RDTAdapter`
- **File**: `models/rdt_adapter.py`
- **Loading**: Direct import
- **Weights**: RDT-1B + T5 + SigLIP (~60GB)
- **Reference**: [RoboticsDiffusionTransformer](https://github.com/RoboUniview/robotics-diffusion-transformer)

### VLA-Touch
- **Class**: `VLA_TouchAdapter`
- **File**: `models/vla_touch_adapter.py`
- **Loading**: Direct import (HuggingFace Transformers)
- **Base**: Qwen2VL + Sparsh tactile encoder
- **Dual registration**: Also registered as TactileAdapter

## Tactile Encoders (TactileAdapter)

### AnyTouch
- **Class**: `AnyTouchAdapter`
- **File**: `models/anytouch_adapter.py`
- **Feature dim**: 768
- **Backbone**: CLIP ViT-L/14
- **Training**: Two-stage: MAE pretrain → CLIP alignment
- **Reference**: [AnyTouch (ICLR 2025)](https://anytouch-spatial-tactile.github.io/)

### Sparsh
- **Class**: `SparshAdapter`
- **File**: `models/sparsh_adapter.py`
- **Feature dim**: 768
- **Backbone**: DINOv2 (ViT-B/14)
- **Weights**: 416MB, available via HuggingFace
- **Verified**: 95.0% linear probe accuracy

### T3
- **Class**: `T3Adapter`
- **File**: `models/t3_adapter.py`
- **Feature dim**: 1024
- **Encoders**: ResNet, CNN, ViT, MAE-ViT (configurable)
- **Notes**: Requires `t3` package

### UniTac_ECF
- **Class**: `UniTac_ECFAdapter`
- **File**: `models/unitac_ecf_adapter.py`
- **Feature dim**: Configurable (delegates to anytouch/sparsh/t3)
- **Runtime delegation**: Uses registry lookup to delegate `encode()` calls
