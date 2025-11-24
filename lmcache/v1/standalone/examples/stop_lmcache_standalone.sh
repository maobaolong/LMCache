#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stop LMCache standalone processes

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping LMCache processes...${NC}"
echo ""

# Stop controller
if [ -f .lmcache_controller.pid ]; then
    CONTROLLER_PID=$(cat .lmcache_controller.pid)
    if kill -0 $CONTROLLER_PID 2>/dev/null; then
        kill $CONTROLLER_PID
        echo -e "${GREEN}✓ Controller Manager stopped (PID: $CONTROLLER_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Controller Manager not running${NC}"
    fi
    rm -f .lmcache_controller.pid
else
    echo -e "${YELLOW}⚠ No controller PID file found${NC}"
fi

# Stop worker
if [ -f .lmcache_worker.pid ]; then
    WORKER_PID=$(cat .lmcache_worker.pid)
    if kill -0 $WORKER_PID 2>/dev/null; then
        kill $WORKER_PID
        echo -e "${GREEN}✓ Standalone Worker stopped (PID: $WORKER_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Worker not running${NC}"
    fi
    rm -f .lmcache_worker.pid
else
    echo -e "${YELLOW}⚠ No worker PID file found${NC}"
fi

echo ""
echo -e "${GREEN}✓ All processes stopped${NC}"
