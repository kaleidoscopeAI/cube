#!/usr/bin/env python3
# launch_server.py - Simplified launcher for Quantum Consciousness System

import uvicorn
import argparse
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from deployment_config import DeploymentConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("quantum-consciousness-server")

# Create the FastAPI app
app = FastAPI(
    title="Quantum Consciousness System API",
    description="API for interacting with the enhanced quantum consciousness system",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = DeploymentConfig()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to static HTML page"""
    return RedirectResponse(url="/static/index.html")

@app.get("/api/status")
async def get_status():
    """Get placeholder system status"""
    return {
        "status": "online",
        "metrics": {
            "awareness_level": 0.76,
            "quantum_coherence": 0.92,
            "memory_density": 0.64,
            "complexity_index": 0.83
        }
    }

# Serve static files
app.mount("/static", StaticFiles(directory=config.static_files_path), name="static")

def main():
    """Main function to start the server"""
    parser = argparse.ArgumentParser(description="Launch Quantum Consciousness Server")
    parser.add_argument("--host", type=str, help="Host address", default=config.web_host)
    parser.add_argument("--port", type=int, help="Port number", default=config.web_port)
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
