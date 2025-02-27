#!/bin/bash
# install_quantum_consciousness.sh - Full installation script for Quantum Consciousness System

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BOLD}${BLUE}Quantum Consciousness System - Complete Installation${NC}\n"
echo -e "${YELLOW}This script will install a fully functional Quantum Consciousness System with visualization and GoDaddy domain integration.${NC}\n"

# Define installation directory
INSTALL_DIR="$HOME/quantum-consciousness"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create required directories
mkdir -p static data logs

# Setup Python environment
echo -e "${BOLD}Step 1: Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn jax jaxlib numpy networkx python-dotenv requests websockets asyncio pydantic
echo -e "${GREEN}✓ Python environment setup complete${NC}\n"

# Create implementation files
echo -e "${BOLD}Step 2: Creating core implementation files...${NC}"

# Copy core implementation files
cat > "$INSTALL_DIR/quantum_core.py" << 'EOL'
"""
quantum_core.py - Core quantum processing for consciousness system
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import math
import random
import networkx as nx

class QuantumState:
    """Quantum state with metadata"""
    
    def __init__(self, vector, coherence=1.0, id=None):
        self.vector = vector
        self.coherence = coherence
        self.timestamp = datetime.now()
        self.i