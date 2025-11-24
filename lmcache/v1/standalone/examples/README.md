# LMCache Standalone Example

This example demonstrates how to use LMCache in standalone mode with LocalCPU and P2P backend support.

## Overview

This example shows a complete LMCache setup optimized for **DeepSeek-V2-Lite** model with:

- **LocalCPU Backend**: 4GB memory cache per worker
- **P2P Backend**: Distributed cache sharing between workers
- **Controller**: Coordinates P2P communication
- **Internal API Server**: Optional REST API for remote access

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LMCache Controller                        │
│         python -m lmcache.v1.api_server                      │
│                  (Port 9000, 5555, 5556)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼──────┐        ┌──────▼───────┐
        │   Worker 0   │◄──────►│   Worker 1   │
        │              │  P2P   │              │
        │ LocalCPU     │        │ LocalCPU     │
        │ 4GB Cache    │        │ 4GB Cache    │
        │ API: 8101    │        │ API: 8102    │
        └──────────────┘        └──────────────┘
```

## Quick Start

### Step 1: Start Controller

```bash
cd /Users/msy/projects/LMCache

python -m lmcache.v1.api_server \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": 5555, "reply": 5556}'
```

Expected output:
```
[INFO] Starting LMCache controller at 0.0.0.0:9000
[INFO] Monitoring lmcache workers at ports {'pull': 5555, 'reply': 5556}
```

### Step 2: Start Worker 0

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config lmcache/v1/standalone/examples/lmcache_standalone_example.yaml \
    --model-name deepseek-ai/DeepSeek-V2-Lite \
    --worker-id 0 \
    --world-size 2 \
    --kv-shape 27,2,256,2,128
```

### Step 3: Start Worker 1

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config lmcache/v1/standalone/examples/lmcache_standalone_example.yaml \
    --model-name deepseek-ai/DeepSeek-V2-Lite \
    --worker-id 1 \
    --world-size 2 \
    --kv-shape 27,2,256,2,128
```

## Configuration

### Configuration File

The example uses `lmcache_standalone_example.yaml`:

```yaml
# Basic settings
chunk_size: 256
cache_policy: "LRU"

# LocalCPU backend
local_cpu: true
max_local_cpu_size: 4.0

# P2P backend
enable_p2p: true
p2p_host: "127.0.0.1"
p2p_init_ports: [6000, 6001]
p2p_lookup_ports: [7000, 7001]

# Controller
enable_controller: true
controller_pull_url: "tcp://127.0.0.1:5555"
controller_reply_url: "tcp://127.0.0.1:5556"

# Internal API server
internal_api_server_enabled: true
internal_api_server_port_start: 8100
```

### Command-Line Parameters

#### Required Parameters

- `--config`: Path to configuration file
- `--model-name`: Model name (e.g., `deepseek-ai/DeepSeek-V2-Lite`)
- `--worker-id`: Worker ID (0, 1, 2, ...)
- `--world-size`: Total number of workers
- `--kv-shape`: KV cache shape as comma-separated integers

#### Optional Parameters

- `--kv-dtype`: KV cache data type (`float16`, `float32`, `bfloat16`)
- `--use-mla`: Enable MLA (Multi-Level Attention)
- `--fmt`: Cache format (default: `vllm`)

#### Extra Parameters (Key-Value Format)

Pass additional parameters using `key=value` format:

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name deepseek-ai/DeepSeek-V2-Lite \
    --worker-id 0 \
    --world-size 2 \
    --kv-shape 27,2,256,2,128 \
    --model-path=/path/to/model \
    --tensor-parallel-size=1 \
    --max-model-len=4096
```

## Model-Specific Configuration

### DeepSeek-V2-Lite

```bash
--model-name deepseek-ai/DeepSeek-V2-Lite
--kv-shape 27,2,256,2,128
--kv-dtype float16
```

**KV Shape Explanation**: `(num_layers, num_kv_heads, chunk_size, num_kv_heads, head_dim)`
- `27`: Number of layers
- `2`: Number of KV heads (GQA)
- `256`: Chunk size (from config)
- `2`: Number of KV heads (repeated)
- `128`: Head dimension

### Other Models

For other models, adjust `--kv-shape` based on model architecture:

```bash
# Example: Llama-3-8B (32 layers, 8 KV heads, 128 head dim)
--kv-shape 32,8,256,8,128

# Example: GPT-2 (12 layers, 12 heads, 64 head dim)
--kv-shape 12,12,256,12,64
```

## Internal API Server

Each worker automatically starts an internal API server (configured in LMCache config file):

- **Worker 0**: Port = `internal_api_server_port_start + 1` (e.g., 8101)
- **Worker 1**: Port = `internal_api_server_port_start + 2` (e.g., 8102)
- **Worker N**: Port = `internal_api_server_port_start + N + 1`

### API Endpoints

Check health:
```bash
curl http://localhost:8101/health
```

Query cache statistics:
```bash
curl http://localhost:8101/stats
```

## P2P Backend Usage

With P2P backend enabled, workers can share cache across nodes:

1. **Worker 0** stores KV cache for tokens 0-100
2. **Worker 1** needs tokens 50-75
3. **Worker 1** queries **Controller** for token location
4. **Controller** returns **Worker 0** address
5. **Worker 1** fetches cache from **Worker 0** via P2P

This enables efficient cache sharing in distributed inference.

## Customization

### Change Number of Workers

Update configuration file:

```yaml
p2p_init_ports: [6000, 6001, 6002, 6003]  # 4 workers
p2p_lookup_ports: [7000, 7001, 7002, 7003]
lmcache_worker_ports: [8000, 8001, 8002, 8003]
```

Start workers:

```bash
for i in {0..3}; do
    python -m lmcache.v1.standalone.lmcache_standalone \
        --config config.yaml \
        --model-name deepseek-ai/DeepSeek-V2-Lite \
        --worker-id $i \
        --world-size 4 \
        --kv-shape 27,2,256,2,128 &
done
```

### Adjust Cache Size

Override in command line:

```bash
--max-local-cpu-size=8.0
```

Or edit configuration file:

```yaml
max_local_cpu_size: 8.0
```

### Disable P2P (CPU-only mode)

Edit configuration:

```yaml
enable_p2p: false
enable_controller: false
```

Start single worker:

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id 0 \
    --world-size 1 \
    --kv-shape 32,8,256,8,128
```

### Add Disk Backend

Edit configuration:

```yaml
local_disk: "file:///tmp/lmcache"
max_local_disk_size: 20.0
```

### Add Remote Backend (Redis)

Edit configuration:

```yaml
remote_url: "redis://localhost:6379"
remote_serde: "naive"
```

## Environment Variables

Set configuration via environment variables:

```bash
export LMCACHE_CONFIG_FILE=/path/to/config.yaml
export LMCACHE_CHUNK_SIZE=512
export LMCACHE_MAX_LOCAL_CPU_SIZE=8.0

python -m lmcache.v1.standalone.lmcache_standalone \
    --model-name my-model \
    --worker-id 0 \
    --world-size 1 \
    --kv-shape 32,8,256,8,128
```

## Troubleshooting

### Issue: "Failed to connect to controller"

**Solution**: Start controller first and verify it's running:

```bash
curl http://localhost:9000/health
```

### Issue: "Port already in use"

**Solution**: Change ports in configuration or kill existing processes:

```bash
lsof -i :5555
kill -9 <PID>
```

### Issue: "Invalid kv_shape format"

**Solution**: Ensure kv_shape matches your model architecture. Format: `num_layers,num_kv_heads,chunk_size,num_kv_heads,head_dim`

### Issue: "Out of memory"

**Solution**: Reduce cache size:

```yaml
max_local_cpu_size: 2.0
```

## Production Deployment

### Using systemd

Create `/etc/systemd/system/lmcache-controller.service`:

```ini
[Unit]
Description=LMCache Controller
After=network.target

[Service]
Type=simple
User=lmcache
WorkingDirectory=/opt/lmcache
ExecStart=/usr/bin/python3 -m lmcache.v1.api_server --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 5555, "reply": 5556}'
Restart=always

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/lmcache-worker@.service`:

```ini
[Unit]
Description=LMCache Worker %i
After=network.target lmcache-controller.service

[Service]
Type=simple
User=lmcache
WorkingDirectory=/opt/lmcache
Environment="LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml"
ExecStart=/usr/bin/python3 -m lmcache.v1.standalone.lmcache_standalone --worker-id %i --world-size 2 --kv-shape 27,2,256,2,128
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
systemctl enable lmcache-controller
systemctl enable lmcache-worker@0
systemctl enable lmcache-worker@1
systemctl start lmcache-controller
systemctl start lmcache-worker@0
systemctl start lmcache-worker@1
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install -e .

CMD ["python", "-m", "lmcache.v1.standalone.lmcache_standalone"]
```

Build and run:

```bash
docker build -t lmcache-standalone .

# Run controller
docker run -d --name lmcache-controller \
    -p 9000:9000 -p 5555:5555 -p 5556:5556 \
    lmcache-standalone \
    python -m lmcache.v1.api_server --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 5555, "reply": 5556}'

# Run worker
docker run -d --name lmcache-worker-0 \
    -v /path/to/config.yaml:/config.yaml \
    -p 8101:8101 \
    lmcache-standalone \
    python -m lmcache.v1.standalone.lmcache_standalone \
    --config /config.yaml \
    --worker-id 0 \
    --world-size 2 \
    --kv-shape 27,2,256,2,128
```

## Files

```
lmcache/v1/standalone/examples/
├── README.md                           # This file
└── lmcache_standalone_example.yaml     # Configuration file
```

## Related Documentation

- [Standalone Module Documentation](../README.md)
- [LMCache Configuration Guide](../../../../docs/configuration.md)

## License

Apache-2.0
