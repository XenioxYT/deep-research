from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import sys
import os
from typing import Dict, Set, Union, Callable, Optional
from datetime import datetime
from pydantic import BaseModel
import logging
import signal

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

# Global application state
class GlobalState:
    def __init__(self):
        self.is_researching: bool = False
        self.current_query: Optional[str] = None
        self.current_report: Optional[str] = None
        self.logs: list = []
        self.chat_history: list = []  # Store follow-up Q&A history
        self.agent: Optional[DeepResearchAgent] = None  # Store agent instance
        self.lock = asyncio.Lock()

    def to_dict(self):
        return {
            "is_researching": self.is_researching,
            "current_query": self.current_query,
            "current_report": self.current_report,
            "logs": self.logs,
            "chat_history": self.chat_history
        }

    async def set_state(self, **kwargs):
        async with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            await manager.broadcast_state_update()

    async def add_log(self, log_entry):
        async with self.lock:
            self.logs.append(log_entry)
            await manager.broadcast_state_update()

global_state = GlobalState()

# Store active research tasks
active_tasks: Dict[str, asyncio.Task] = {}

# Request model
class ResearchRequest(BaseModel):
    query: str
    client_id: str

class FollowupRequest(BaseModel):
    question: str
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
            # Send current state to the new client
            await self.send_state_to_client(websocket)

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def send_state_to_client(self, websocket: WebSocket):
        """Send current application state to a specific client."""
        try:
            await websocket.send_text(json.dumps({
                "type": "state_update",
                "data": global_state.to_dict()
            }))
        except Exception:
            pass

    async def broadcast_state_update(self):
        """Broadcast current state to all connected clients."""
        state_message = json.dumps({
            "type": "state_update",
            "data": global_state.to_dict()
        })
        
        for client_connections in self.active_connections.values():
            for websocket in set(client_connections):
                try:
                    await websocket.send_text(state_message)
                except Exception:
                    continue

    async def broadcast_to_client(self, message: str, client_id: str):
        """Send a message to all WebSocket connections for a specific client."""
        if client_id in self.active_connections:
            connections = set(self.active_connections[client_id])
            disconnected_sockets = set()
            
            for websocket in connections:
                try:
                    await websocket.send_text(message)
                except Exception:
                    disconnected_sockets.add(websocket)

            if disconnected_sockets:
                async with self._lock:
                    if client_id in self.active_connections:
                        self.active_connections[client_id] -= disconnected_sockets
                        if not self.active_connections[client_id]:
                            del self.active_connections[client_id]

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
        # Add to global state
        await global_state.add_log(log_entry)
        # Also send directly to the client for immediate feedback
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
    # Check if research is already in progress
    if global_state.is_researching:
        return JSONResponse(
            content={"status": "error", "message": "Research is already in progress"},
            status_code=400
        )

    try:
        # Update global state
        await global_state.set_state(
            is_researching=True,
            current_query=request.query,
            current_report=None,
            logs=[],
            chat_history=[]  # Reset chat history for new research
        )

        # Create hybrid logger
        logger = HybridLogger(request.client_id)
        logger.info(f"Starting research for query: {request.query}")

        # Create and store the research task
        async def research_task():
            try:
                async with DeepResearchAgent() as agent:
                    # Store agent instance for follow-up questions
                    global_state.agent = agent
                    agent.logger = logger
                    report_path = await agent.research(request.query)
                    
                    if report_path.startswith("Report has been generated and saved to: "):
                        report_path = report_path.replace("Report has been generated and saved to: ", "")
                        try:
                            with open(report_path, 'r', encoding='utf-8') as f:
                                report_content = f.read()
                        except Exception as e:
                            logger.error(f"Error reading report file: {str(e)}")
                            report_content = "Error: Could not read report content."
                    else:
                        report_content = report_path

                    logger.info("Research completed")
                    await global_state.set_state(
                        is_researching=False,
                        current_report=report_content
                    )
                    return {"status": "success", "result": report_content}
            except Exception as e:
                logger.error(f"Error during research: {str(e)}")
                await global_state.set_state(is_researching=False)
                return {"status": "error", "message": str(e)}

        task = asyncio.create_task(research_task())
        active_tasks[request.client_id] = task
        result = await task

        # Clean up task
        if request.client_id in active_tasks:
            del active_tasks[request.client_id]

        return JSONResponse(content=result)

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error during research: {str(e)}")
        await global_state.set_state(is_researching=False)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Follow-up question endpoint
@app.post("/api/followup")
async def followup_question(request: FollowupRequest):
    # Check if we have a report and agent to work with
    if not global_state.current_report or not global_state.agent:
        return JSONResponse(
            content={"status": "error", "message": "No research report available to answer follow-up questions"},
            status_code=400
        )

    try:
        # Create hybrid logger
        logger = HybridLogger(request.client_id)
        logger.info(f"Processing follow-up question: {request.question}")
        
        # Set the agent's logger
        global_state.agent.logger = logger
        
        # Create task for answering the follow-up question
        async def followup_task():
            try:
                # Answer the follow-up question
                answer = await global_state.agent.answer_followup_question(
                    request.question, 
                    global_state.current_query, 
                    global_state.current_report
                )
                
                # Add to chat history
                chat_entry = {
                    "question": request.question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update global state with new chat entry
                async with global_state.lock:
                    global_state.chat_history.append(chat_entry)
                
                # Broadcast state update to clients
                await manager.broadcast_state_update()
                
                logger.info("Follow-up question answered successfully")
                return {"status": "success", "result": answer}
            except Exception as e:
                logger.error(f"Error answering follow-up question: {str(e)}")
                return {"status": "error", "message": str(e)}
        
        # Execute the follow-up task
        task = asyncio.create_task(followup_task())
        active_tasks[request.client_id] = task
        result = await task
        
        # Clean up task
        if request.client_id in active_tasks:
            del active_tasks[request.client_id]
            
        return JSONResponse(content=result)
    
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error processing follow-up question: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Get current state endpoint
@app.get("/api/state")
async def get_state():
    return JSONResponse(content=global_state.to_dict())

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 