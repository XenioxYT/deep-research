from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import sys
import os
from typing import Dict, Set, Union, Callable
from datetime import datetime
from pydantic import BaseModel
import logging

# Add parent directory to path to import DeepResearchAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from deep_research_agent import DeepResearchAgent

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8992",
        "http://127.0.0.1:8992",
        "http://100.67.165.122:8992",
        "http://0.0.0.0:8992",
        "http://192.168.0.15:8992",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ResearchRequest(BaseModel):
    query: str
    client_id: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        async with self._lock:
            if client_id not in self.active_connections:
                self.active_connections[client_id] = set()
            self.active_connections[client_id].add(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def broadcast_to_client(self, message: str, client_id: str):
        if client_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_text(message)
                except:
                    dead_connections.add(connection)
            
            if dead_connections:
                async with self._lock:
                    for dead in dead_connections:
                        self.active_connections[client_id].discard(dead)

manager = ConnectionManager()

# Hybrid logger that works with both sync and async code
class HybridLogger(logging.Logger):
    def __init__(self, client_id: str):
        super().__init__(name=f"hybrid_logger_{client_id}")
        self.client_id = client_id
        self.loop = asyncio.get_event_loop()

    async def _async_log(self, level: str, message: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        await manager.broadcast_to_client(
            json.dumps({"type": "log", "data": log_entry}),
            self.client_id
        )

    def _sync_log(self, level: str, message: str):
        if not self.loop.is_running():
            self.loop = asyncio.new_event_loop()
        asyncio.run_coroutine_threadsafe(
            self._async_log(level, message),
            self.loop
        )

    def log(self, level: str, message: str):
        try:
            if asyncio.get_running_loop():
                asyncio.create_task(self._async_log(level, message))
            else:
                self._sync_log(level, message)
        except RuntimeError:
            self._sync_log(level, message)

    def info(self, message: str):
        self.log("info", message)

    def warning(self, message: str):
        self.log("warning", message)

    def error(self, message: str):
        self.log("error", message)

    def debug(self, message: str):
        self.log("debug", message)

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

# Research endpoint
@app.post("/api/research")
async def start_research(request: ResearchRequest):
    try:
        # Create hybrid logger
        logger = HybridLogger(request.client_id)
        logger.info(f"Starting research for query: {request.query}")

        # Initialize research agent with hybrid logger
        async with DeepResearchAgent() as agent:
            # Replace agent's logger with hybrid logger
            agent.logger = logger
            
            # Start research
            result = await agent.research(request.query)
            
            # Send completion message
            logger.info("Research completed")
            
            return JSONResponse(
                content={"status": "success", "result": result},
                status_code=200
            )
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error during research: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 