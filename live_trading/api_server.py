"""FastAPI server with WebSocket/SSE for real-time dashboard updates."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


class DashboardServer:
    """FastAPI server for live trading dashboard."""
    
    def __init__(self, trading_engine: Any, config: Any):
        """Initialize dashboard server.
        
        Args:
            trading_engine: Reference to trading engine
            config: Trading configuration
        """
        self.trading_engine = trading_engine
        self.config = config
        
        self.app = FastAPI(title="Live Trading Dashboard")
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Background task for broadcasting updates
        self.broadcast_task = None
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the dashboard HTML."""
            from pathlib import Path
            dashboard_path = Path(__file__).parent / "dashboard.html"
            
            if dashboard_path.exists():
                with open(dashboard_path, 'r') as f:
                    return f.read()
            else:
                return "<h1>Dashboard file not found</h1>"
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            engine_state = self.trading_engine.get_state()
            perf_metrics = self.trading_engine.performance_tracker.get_metrics()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'engine': engine_state,
                'metrics': perf_metrics,
                'config': {
                    'symbol': self.config['symbol'],
                    'mode': self.config['mode'],
                    'timeframe': self.config['timeframe'],
                    'strategy': self.config['strategy']['name'],
                }
            }
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get performance metrics."""
            return self.trading_engine.performance_tracker.get_metrics()
        
        @self.app.get("/api/equity_curve")
        async def get_equity_curve():
            """Get equity curve data."""
            equity_df = self.trading_engine.performance_tracker.get_equity_curve()
            
            if equity_df.empty:
                return {'timestamps': [], 'equity': [], 'capital': [], 'unrealized_pnl': []}
            
            return {
                'timestamps': [ts.isoformat() for ts in equity_df['timestamp']],
                'equity': equity_df['equity'].tolist(),
                'capital': equity_df['capital'].tolist(),
                'unrealized_pnl': equity_df['unrealized_pnl'].tolist(),
            }
        
        @self.app.get("/api/trades")
        async def get_trades():
            """Get trade history."""
            trades_df = self.trading_engine.performance_tracker.get_trades_df()
            
            if trades_df.empty:
                return {'trades': []}
            
            # Convert to list of dicts
            trades_list = trades_df.to_dict('records')
            
            # Convert timestamps to ISO format
            for trade in trades_list:
                if 'timestamp' in trade and isinstance(trade['timestamp'], datetime):
                    trade['timestamp'] = trade['timestamp'].isoformat()
            
            return {'trades': trades_list}
        
        @self.app.get("/api/orderbook")
        async def get_orderbook():
            """Get current orderbook snapshot."""
            if not self.trading_engine.orderbook_streamer:
                return {'snapshot': None}
            
            snapshot = self.trading_engine.orderbook_streamer.get_current_snapshot()
            imbalance = self.trading_engine.orderbook_streamer.get_imbalance(
                percentage=self.config.get('strategy.params.percentage', 1)
            )
            
            return {
                'snapshot': snapshot,
                'imbalance': imbalance,
                'timestamp': datetime.utcnow().isoformat(),
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
        
        @self.app.get("/api/stream")
        async def stream_updates():
            """Server-Sent Events endpoint for updates."""
            async def event_generator():
                while True:
                    # Get current state
                    state = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'engine': self.trading_engine.get_state(),
                        'metrics': self.trading_engine.performance_tracker.get_metrics(),
                    }
                    
                    yield f"data: {json.dumps(state)}\n\n"
                    await asyncio.sleep(1)  # Update every second
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )
    
    async def broadcast_update(self, data: Dict[str, Any]) -> None:
        """Broadcast update to all connected WebSocket clients.
        
        Args:
            data: Data to broadcast
        """
        if not self.active_connections:
            return
        
        message = json.dumps(data)
        
        # Send to all connections
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    async def start_broadcasting(self) -> None:
        """Start background task to broadcast updates."""
        while True:
            try:
                # Gather current state
                update = {
                    'type': 'update',
                    'timestamp': datetime.utcnow().isoformat(),
                    'engine': self.trading_engine.get_state(),
                    'metrics': self.trading_engine.performance_tracker.get_metrics(),
                }
                
                # Broadcast to all clients
                await self.broadcast_update(update)
                
                # Wait before next update
                await asyncio.sleep(1)
            
            except Exception as e:
                print(f"[DashboardServer] Error broadcasting: {e}")
                await asyncio.sleep(5)
    
    async def run(self) -> None:
        """Run the FastAPI server."""
        host = self.config.get('server.host', '0.0.0.0')
        port = self.config.get('server.port', 8000)
        
        # Start broadcasting task
        self.broadcast_task = asyncio.create_task(self.start_broadcasting())
        
        print(f"[DashboardServer] Starting on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self) -> None:
        """Stop the server."""
        if self.broadcast_task:
            self.broadcast_task.cancel()
        
        # Close all connections
        for connection in self.active_connections:
            await connection.close()
        
        self.active_connections.clear()

