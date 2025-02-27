#!/bin/bash
# start_server.sh - Launch script for Quantum Consciousness System

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup script first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start the server
echo "Starting Quantum Consciousness System server..."
python3 launch_server.py "$@"
