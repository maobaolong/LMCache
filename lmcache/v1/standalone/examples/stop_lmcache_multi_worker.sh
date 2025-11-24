#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stop LMCache Multi-Worker processes

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping LMCache Multi-Worker processes...${NC}"
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

# Stop worker 0
if [ -f .lmcache_worker0.pid ]; then
    WORKER0_PID=$(cat .lmcache_worker0.pid)
    if kill -0 $WORKER0_PID 2>/dev/null; then
        kill $WORKER0_PID
        echo -e "${GREEN}✓ Worker 0 stopped (PID: $WORKER0_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Worker 0 not running${NC}"
    fi
    rm -f .lmcache_worker0.pid
else
    echo -e "${YELLOW}⚠ No worker 0 PID file found${NC}"
fi

# Stop worker 1
if [ -f .lmcache_worker1.pid ]; then
    WORKER1_PID=$(cat .lmcache_worker1.pid)
    if kill -0 $WORKER1_PID 2>/dev/null; then
        kill $WORKER1_PID
        echo -e "${GREEN}✓ Worker 1 stopped (PID: $WORKER1_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Worker 1 not running${NC}"
    fi
    rm -f .lmcache_worker1.pid
else
    echo -e "${YELLOW}⚠ No worker 1 PID file found${NC}"
fi

echo ""
echo -e "${GREEN}✓ All processes stopped${NC}"
