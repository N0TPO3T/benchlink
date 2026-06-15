# Inference Server Protocol

## JSON Lines Protocol

BenchLink uses a simple JSON Lines protocol for communication between ModelAdapters and inference servers running in Docker containers or subprocesses.

### Protocol Flow

```
Adapter                              Server
  │                                     │
  │  ── {"status": "ready"} ──────────  │  (server sends on startup)
  │                                     │
  │  ── {"obs": {...}, "id": 0} ──────  │  (inference request)
  │                                     │
  │  ── {"action": [...], "id": 0} ───  │  (inference response)
  │                                     │
  │  ── {"reset": true, "id": 1} ─────  │  (reset episode)
  │                                     │
  │  ── {"stop": true, "id": 2} ──────  │  (shutdown)
```

### Message Formats

**Ready Signal** (server → adapter):
```json
{"status": "ready"}
```

**Inference Request** (adapter → server):
```json
{
  "obs": {
    "rgb_static": [[...]],
    "rgb_gripper": [[...]],
    "proprio": [0.1, 0.2, ...],
    "language": "pick up the object",
    "tactile_img": [[...]]
  },
  "id": 0
}
```

**Inference Response** (server → adapter):
```json
{
  "action": [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.5],
  "id": 0
}
```

**Reset Signal** (adapter → server):
```json
{"reset": true, "id": 1}
```

**Stop Signal** (adapter → server):
```json
{"stop": true, "id": 2}
```

## Server Implementations

### RDP Server
- **File**: `servers/rdp_server.py`
- **Suggested name**: `rdp_server`
- **Prefix**: `__RDP__` (to filter model log output from JSON responses)
- **Action format**: 4-D TCP absolute position → converted to 7-D delta pose

### OpenPI Server
- **File**: `servers/openpi_server.py`
- **Suggested name**: `openpi_server`
- **Fallbacks**: octopi → pi0 → raw torch.load
- **Features**: Graceful fallback chain for model loading

### Motus Server
- **File**: `servers/motus_server.py`
- **Suggested name**: `motus_server`
- **Models**: Wan2.2-TI2V-5B, T5 encoder, Qwen3-VL
- **Features**: Language cache with LRU eviction

### DreamZero Server
- **File**: `servers/dreamzero_server.py`
- **Suggested name**: `dreamzero_server`
- **Models**: VLA.from_pretrained (16.5B INT8)
- **Features**: Latent history buffer for video prediction
- **Paths**: `DREAMZERO_ROOT` and `CHECKPOINT_DIR` configurable via CLI args

## Docker Adapter Lifecycle

```
1. Adapter.load() checks if container exists
   → _connect_existing_container() or _ensure_container()

2. _start_server() runs inside container:
   docker exec -i <container> python <server_script>

3. Server sends {"status": "ready"} on stdout

4. Each model.act(obs) → _send_request({"obs": obs, "id": N})
   → writes JSON line to stdin
   → reads JSON line from stdout
   → returns action array

5. model.reset() → _send_request({"reset": true})

6. model.close() → _send_request({"stop": true}) → terminates subprocess
```
