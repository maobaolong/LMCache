#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Quick start script for LMCache Multi-Worker with Controller Manager

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LMCache Multi-Worker Quick Start${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if config file exists
CONFIG_FILE="lmcache/v1/standalone/examples/lmcache_standalone_multi_worker.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Starting Controller Manager...${NC}"
python -m lmcache.v1.api_server \
    --host 0.0.0.0 \
    --port 9000 \
    --monitor-ports '{"pull": "5555", "reply": "5556"}' \
    > controller.log 2>&1 &

CONTROLLER_PID=$!
echo -e "${GREEN}✓ Controller Manager started (PID: $CONTROLLER_PID)${NC}"
echo -e "  API: http://0.0.0.0:9000"
echo -e "  Logs: controller.log"
echo ""

# Wait for controller to start
echo -e "${YELLOW}Waiting for Controller Manager to be ready...${NC}"
sleep 3

# Check if controller is running
if ! kill -0 $CONTROLLER_PID 2>/dev/null; then
    echo -e "${RED}✗ Controller Manager failed to start${NC}"
    echo -e "${RED}Check controller.log for details${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 2: Starting Worker 0...${NC}"
export LMCACHE_CONFIG_FILE="$CONFIG_FILE"
python -m lmcache.v1.standalone.lmcache_standalone \
    --worker-id 0 \
    --world-size 2 \
    > worker0.log 2>&1 &

WORKER0_PID=$!
echo -e "${GREEN}✓ Worker 0 started (PID: $WORKER0_PID)${NC}"
echo -e "  API: http://0.0.0.0:8100"
echo -e "  Logs: worker0.log"
echo ""

echo -e "${YELLOW}Step 3: Starting Worker 1...${NC}"
export LMCACHE_CONFIG_FILE="$CONFIG_FILE"
python -m lmcache.v1.standalone.lmcache_standalone \
    --worker-id 1 \
    --world-size 2 \
    > worker1.log 2>&1 &

WORKER1_PID=$!
echo -e "${GREEN}✓ Worker 1 started (PID: $WORKER1_PID)${NC}"
echo -e "  API: http://0.0.0.0:8101"
echo -e "  Logs: worker1.log"
echo ""

# Wait for workers to register
echo -e "${YELLOW}Waiting for workers to register...${NC}"
sleep 5

# Check if workers are running
for pid in $WORKER0_PID $WORKER1_PID; do
    if ! kill -0 $pid 2>/dev/null; then
        echo -e "${RED}✗ Worker failed to start${NC}"
        echo -e "${RED}Check worker logs for details${NC}"
        kill $CONTROLLER_PID 2>/dev/null || true
        exit 1
    fi
done

echo -e "${GREEN}✓ All workers are running${NC}"
echo ""

echo -e "${YELLOW}Step 4: Verifying registration...${NC}"
sleep 2

# Query workers
WORKERS=$(curl -s http://0.0.0.0:9000/controller/workers 2>/dev/null || echo '{"total_count": 0}')
WORKER_COUNT=$(echo "$WORKERS" | grep -o '"total_count":[0-9]*' | grep -o '[0-9]*' || echo "0")

# Ensure WORKER_COUNT is not empty
WORKER_COUNT=${WORKER_COUNT:-0}

if [ "$WORKER_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Workers registered successfully!${NC}"
    echo -e "${GREEN}  Total workers: $WORKER_COUNT${NC}"
else
    echo -e "${YELLOW}⚠ Workers not registered yet (may take a few more seconds)${NC}"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LMCache Multi-Worker is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Controller Manager:"
echo -e "  PID: $CONTROLLER_PID"
echo -e "  API: ${GREEN}http://0.0.0.0:9000${NC}"
echo -e "  Logs: controller.log"
echo ""
echo -e "Worker 0:"
echo -e "  PID: $WORKER0_PID"
echo -e "  API: ${GREEN}http://0.0.0.0:8100${NC}"
echo -e "  Logs: worker0.log"
echo ""
echo -e "Worker 1:"
echo -e "  PID: $WORKER1_PID"
echo -e "  API: ${GREEN}http://0.0.0.0:8101${NC}"
echo -e "  Logs: worker1.log"
echo ""
echo -e "Useful commands:"
echo -e "  ${YELLOW}# Query all workers${NC}"
echo -e "  curl http://0.0.0.0:9000/controller/workers"
echo ""
echo -e "  ${YELLOW}# Query cluster status${NC}"
echo -e "  curl http://0.0.0.0:9000/controller/cluster/status"
echo ""
echo -e "  ${YELLOW}# Query worker 0 metrics${NC}"
echo -e "  curl http://0.0.0.0:8100/metrics"
echo ""
echo -e "  ${YELLOW}# Query worker 1 metrics${NC}"
echo -e "  curl http://0.0.0.0:8101/metrics"
echo ""
echo -e "  ${YELLOW}# Stop all processes${NC}"
echo -e "  kill $CONTROLLER_PID $WORKER0_PID $WORKER1_PID"
echo ""
echo -e "  ${YELLOW}# View logs${NC}"
echo -e "  tail -f controller.log"
echo -e "  tail -f worker0.log"
echo -e "  tail -f worker1.log"
echo ""

# Save PIDs to file for easy cleanup
echo "$CONTROLLER_PID" > .lmcache_controller.pid
echo "$WORKER0_PID" > .lmcache_worker0.pid
echo "$WORKER1_PID" > .lmcache_worker1.pid

echo -e "${GREEN}Process IDs saved to .lmcache_*.pid files${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop (or run: kill \$(cat .lmcache_*.pid))${NC}"
echo ""

# Wait for user interrupt
trap "echo ''; echo -e '${YELLOW}Stopping LMCache...${NC}'; kill $CONTROLLER_PID $WORKER0_PID $WORKER1_PID 2>/dev/null || true; rm -f .lmcache_*.pid; echo -e '${GREEN}✓ Stopped${NC}'; exit 0" INT TERM

# Keep script running
wait
