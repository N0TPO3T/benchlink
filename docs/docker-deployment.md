# Docker Deployment Guide

## Overview

Some models (OpenPI, Motus, DreamZero) require Docker containers due to:
- Heavy dependencies (CUDA, specific PyTorch versions)
- Hardware requirements (multi-GPU setups)
- Environment isolation

BenchLink provides a `DockerModelAdapter` base class that handles container lifecycle and communication.

## Deployment Steps

### 1. Build or Pull the Docker Image

```bash
# Example for a DreamZero container
docker pull <your-registry>/dreamzero:l2-code

# Or build from Dockerfile
docker build -t dreamzero:l2-code -f deploy/Dockerfile.dreamzero .
```

### 2. Copy the Inference Server

```bash
# Deploy the inference server script into the container
./deploy/deploy_inference_servers.sh --adapter dreamzero
```

### 3. Run Evaluation

```bash
benchlink --model dreamzero --benchmark maniskill \
  --checkpoint "" \
  --config configs/default.yaml \
  --episodes 5
```

## Container Requirements

Each Docker container needs:
1. A running Python environment with the model dependencies
2. The inference server script copied to `/workspace/`
3. GPU access (`--gpus all` or `--gpus device=0`)
4. Sufficient shared memory (`--shm-size 16G`)

## Deploy Script

The `deploy_inference_servers.sh` script automates:
- Copying server scripts to containers via `docker cp`
- Verifying scripts start correctly
- Running a smoke test

```bash
# Deploy all servers
./deploy/deploy_inference_servers.sh --host <server-ip>

# Deploy and test a specific adapter
./deploy/deploy_inference_servers.sh --adapter openpi --test
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | Server not started | Check `docker ps`; verify `_start_server()` timeout |
| `{"error": ...}` | Invalid request format | Check `_build_request()` output matches expected JSON |
| Action shape mismatch | Dimension mismatch | Verify `_from_canonical` / `_clip_action` config |
