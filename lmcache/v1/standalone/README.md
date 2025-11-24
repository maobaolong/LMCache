# LMCache Standalone Module

A production-ready standalone module for LMCacheEngine that works without vLLM or GPU.

## Overview

The LMCache Standalone module enables you to run LMCacheEngine independently for distributed KV cache storage and sharing via P2P backend. This is useful when you want to:

- **Run dedicated cache nodes** without inference workload
- **Share KV cache across multiple inference nodes** via P2P
- **Test LMCache functionality** without vLLM dependencies
- **Build custom caching solutions** on top of LMCache

## Features

- ✅ **Standalone Operation**: No vLLM or GPU required
- ✅ **All Backend Support**: CPU, Disk, P2P, Remote
- ✅ **P2P Cache Sharing**: Workers can fetch cache from each other via Controller
- ✅ **Internal API Server**: Optional REST API for remote access
- ✅ **Flexible Configuration**: YAML file + CLI overrides + environment variables
- ✅ **Production Ready**: Signal handling, logging, graceful shutdown

## Installation

```bash
pip install lmcache
```

Or from source:

```bash
cd /path/to/LMCache
pip install -e .
```

## Quick Start

### 1. CPU-Only Mode (Simplest)

```bash
cat > config.yaml << EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 2.0
enable_p2p: false
enable_controller: false
lmcache_instance_id: "my_lmcache"
EOF

python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id 0 \
    --world-size 1 \
    --kv-shape 32,8,256,8,128
```

### 2. P2P Distributed Mode (Recommended)

**Terminal 1: Start Controller**

```bash
python -m lmcache.v1.api_server \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": 5555, "reply": 5556}'
```

**Terminal 2: Start Worker 0**

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id 0 \
    --world-size 2 \
    --kv-shape 32,8,256,8,128
```

**Terminal 3: Start Worker 1**

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id 1 \
    --world-size 2 \
    --kv-shape 32,8,256,8,128
```

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

## Configuration

### Configuration File (YAML)

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

# Instance ID
lmcache_instance_id: "my_lmcache"
```

### Command-Line Parameters

#### Required Parameters

```bash
--config <path>              # Path to configuration file
--model-name <name>          # Model name for cache identification
--worker-id <id>             # Worker ID (0, 1, 2, ...)
--world-size <size>          # Total number of workers
--kv-shape <shape>           # KV cache shape (comma-separated)
```

#### Optional Parameters

```bash
--kv-dtype <dtype>           # float16, float32, bfloat16 (default: float16)
--use-mla                    # Enable MLA (Multi-Level Attention)
--fmt <format>               # Cache format (default: vllm)
```

#### Extra Parameters (Key-Value)

Pass additional parameters using `key=value` format:

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id 0 \
    --world-size 1 \
    --kv-shape 32,8,256,8,128 \
    --custom-param1=value1 \
    --custom-param2=value2
```

### Configuration Priority

1. **Command-line arguments** (highest)
2. **Configuration file**
3. **Environment variables**
4. **Default values** (lowest)

### Environment Variables

```bash
export LMCACHE_CONFIG_FILE=/path/to/config.yaml
export LMCACHE_CHUNK_SIZE=512
export LMCACHE_MAX_LOCAL_CPU_SIZE=8.0
```

## KV Shape Configuration

The `--kv-shape` parameter defines the KV cache tensor shape:

**Format**: `num_layers,num_kv_heads,chunk_size,num_kv_heads,head_dim`

### Examples

**DeepSeek-V2-Lite** (27 layers, 2 KV heads, 128 head dim):
```bash
--kv-shape 27,2,256,2,128
```

**Llama-3-8B** (32 layers, 8 KV heads, 128 head dim):
```bash
--kv-shape 32,8,256,8,128
```

**GPT-2** (12 layers, 12 heads, 64 head dim):
```bash
--kv-shape 12,12,256,12,64
```

**Note**: The `chunk_size` (3rd dimension) should match the `chunk_size` in your configuration file.

## Internal API Server

Each worker automatically starts an internal API server (configured in LMCache config file):

- **Worker 0**: Port = `internal_api_server_port_start + 1` (e.g., 8101)
- **Worker 1**: Port = `internal_api_server_port_start + 2` (e.g., 8102)
- **Worker N**: Port = `internal_api_server_port_start + N + 1`

The internal API server is always enabled and its configuration is read from the LMCache config file:

```yaml
internal_api_server_enabled: true
internal_api_server_host: "0.0.0.0"
internal_api_server_port_start: 8100
```

### API Endpoints

```bash
# Health check (assuming port 8101 for worker 0)
curl http://localhost:8101/health

# Cache statistics
curl http://localhost:8101/stats
```

## P2P Backend Usage

With P2P backend enabled, workers can share cache across nodes:

1. **Worker 0** stores KV cache for tokens 0-100
2. **Worker 1** needs tokens 50-75
3. **Worker 1** queries **Controller** for token location
4. **Controller** returns **Worker 0** address
5. **Worker 1** fetches cache from **Worker 0** via P2P

This enables efficient cache sharing in distributed inference scenarios.

## Starting the Controller

The controller is started separately using the existing `lmcache.v1.api_server` module:

```bash
python -m lmcache.v1.api_server \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": 5555, "reply": 5556}'
```

**Parameters**:
- `--host`: Controller host (default: 0.0.0.0)
- `--port`: Controller API port (default: 9000)
- `--monitor-ports`: JSON string with pull and reply ports

## Examples

See the [examples directory](examples/) for complete examples:

- **DeepSeek-V2-Lite**: Complete setup with LocalCPU + P2P
- **Configuration file**: `lmcache_standalone_example.yaml`
- **Detailed documentation**: [examples/README.md](examples/README.md)

## Use Cases

### 1. Dedicated Cache Nodes

Run standalone LMCache workers as dedicated cache nodes:

```bash
# Node 1: Cache worker
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name my-model \
    --worker-id $i \
    --world-size 4 \
    --kv-shape 32,8,256,8,128 &
done

# Node 2: Inference worker (vLLM)
# Configure vLLM to use P2P backend and connect to controller
```

### 2. Multi-Node Cache Sharing

Deploy workers across multiple nodes:

```bash
# Node 1: Controller
python -m lmcache.v1.api_server --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 5555, "reply": 5556}'

# Node 2: Worker 0
python -m lmcache.v1.standalone.lmcache_standalone --config config.yaml --worker-id 0 --world-size 4 --kv-shape 32,8,256,8,128

# Node 3: Worker 1
python -m lmcache.v1.standalone.lmcache_standalone --config config.yaml --worker-id 1 --world-size 4 --kv-shape 32,8,256,8,128

# Node 4: Worker 2
python -m lmcache.v1.standalone.lmcache_standalone --config config.yaml --worker-id 2 --world-size 4 --kv-shape 32,8,256,8,128

# Node 5: Worker 3
python -m lmcache.v1.standalone.lmcache_standalone --config config.yaml --worker-id 3 --world-size 4 --kv-shape 32,8,256,8,128
```

### 3. Testing and Development

Test LMCache functionality without vLLM:

```bash
python -m lmcache.v1.standalone.lmcache_standalone \
    --config config.yaml \
    --model-name test-model \
    --worker-id 0 \
    --world-size 1 \
    --kv-shape 12,12,256,12,64
```

## Troubleshooting

### "Failed to connect to controller"

Start controller first:
```bash
python -m lmcache.v1.api_server --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 5555, "reply": 5556}'
```

### "Port already in use"

Change ports in configuration or kill existing processes:
```bash
lsof -i :5555
kill -9 <PID>
```

### "Invalid kv_shape format"

Ensure format is correct: `num_layers,num_kv_heads,chunk_size,num_kv_heads,head_dim`

### "Out of memory"

Reduce cache size in configuration:
```yaml
max_local_cpu_size: 2.0
```
