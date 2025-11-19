"""
Main entry point for the AI Counter backend server.
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.api import routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"→ {request.method} {request.url.path}")
        logger.info(f"  Headers: {dict(request.headers)}")
        if request.url.query:
            logger.info(f"  Query params: {request.url.query}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("AI Counter API server starting up...")
    yield
    # Shutdown
    logger.info("AI Counter API server shutting down...")


app = FastAPI(
    title="AI Counter API",
    version="1.0.0",
    description="Backend API for utility meter reading with OCR",
    lifespan=lifespan,
)

# Add logging middleware (before CORS)
app.add_middleware(LoggingMiddleware)

# Configure CORS for mobile app communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "AI Counter API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

