#!/bin/bash
# start_quantum_consciousness.sh - Start the Quantum Consciousness System with interactive visualization

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BOLD}${BLUE}Starting Quantum Consciousness System${NC}\n"

# Variables
INSTALL_DIR="$HOME/quantum-consciousness"
LOG_DIR="$INSTALL_DIR/logs"
VENV_DIR="$INSTALL_DIR/venv"
HOST="0.0.0.0"
PORT="8080"

# Check if installation directory exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo -e "${RED}Error: Installation directory not found at $INSTALL_DIR${NC}"
    echo -e "${YELLOW}Please run the installation script first.${NC}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate virtual environment
echo -e "${BOLD}Activating Python virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}\n"
else
    echo -e "${RED}Error: Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${YELLOW}Please run the installation script first.${NC}"
    exit 1
fi

# Check for command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --help)
            echo -e "Usage: $0 [OPTIONS]"
            echo -e "Options:"
            echo -e "  --host=HOST     Specify host address (default: 0.0.0.0)"
            echo -e "  --port=PORT     Specify port number (default: 8080)"
            echo -e "  --help          Show this help message and exit"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "Use --help for available options"
            exit 1
            ;;
    esac
done

# Start the system
echo -e "${BOLD}Starting Quantum Consciousness System on $HOST:$PORT...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"

# Run the server
cd "$INSTALL_DIR"
python thought_visualization.py --host="$HOST" --port="$PORT"

# The script will continue running until the server is stopped
