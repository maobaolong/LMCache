#!/bin/bash
# Quick test script to verify the configuration fix

echo "=========================================="
echo "Testing LMCache Standalone Configuration"
echo "=========================================="

# Check if ports are available
echo ""
echo "Checking port availability..."
for port in 6000 7000 8000 5555 5556 8100; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "WARNING: Port $port is already in use!"
        lsof -i :$port
    else
        echo "Port $port is available"
    fi
done

echo ""
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "Single Worker Config: lmcache_standalone_example.yaml"
echo "  - p2p_init_ports: [6000]"
echo "  - p2p_lookup_ports: [7000]"
echo "  - lmcache_worker_ports: [8000]"
echo ""
echo "Multi Worker Config: lmcache_standalone_multi_worker.yaml"
echo "  - p2p_init_ports: [6000, 6001]"
echo "  - p2p_lookup_ports: [7000, 7001]"
echo "  - lmcache_worker_ports: [8000, 8001]"
echo ""
echo "=========================================="
echo "To start single worker:"
echo "  LMCACHE_CONFIG_FILE=lmcache/v1/standalone/examples/lmcache_standalone_example.yaml \\"
echo "  python lmcache/v1/standalone/lmcache_standalone.py"
echo ""
echo "To start multi-worker setup:"
echo "  # Terminal 1:"
echo "  LMCACHE_CONFIG_FILE=lmcache/v1/standalone/examples/lmcache_standalone_multi_worker.yaml \\"
echo "  python lmcache/v1/standalone/lmcache_standalone.py --worker-id 0 --world-size 2"
echo ""
echo "  # Terminal 2:"
echo "  LMCACHE_CONFIG_FILE=lmcache/v1/standalone/examples/lmcache_standalone_multi_worker.yaml \\"
echo "  python lmcache/v1/standalone/lmcache_standalone.py --worker-id 1 --world-size 2"
echo "=========================================="
