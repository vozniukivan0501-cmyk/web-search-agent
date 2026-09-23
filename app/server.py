"""FastAPI web server with SSE streaming."""
import logging
import json
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent import run_agent, run_agent_stream
from app.cross_encoder import get_cross_encoder
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load cross-encoder model on startup."""
    logger.info("Loading Cross-Encoder model on startup...")
    get_cross_encoder(settings.cross_encoder_model)
    logger.info("Application ready!")
    yield


app = FastAPI(
    title="AI Web Search Agent",
    description="Multi-step reasoning web search agent powered by Gemini, LlamaIndex, and LangGraph",
    version="1.0.0",
    lifespan=lifespan
)


class SearchRequest(BaseModel):
    query: str
    api_key: Optional[str] = None


def resolve_api_key(user_key: Optional[str]) -> str:
    """Resolve API key: use user provided key if present, otherwise fallback to server default."""
    if user_key and user_key.strip():
        return user_key.strip()
    if settings.google_api_key and settings.google_api_key.strip():
        return settings.google_api_key.strip()
    raise HTTPException(
        status_code=400,
        detail="API key is required. Please enter a Google AI API key or set GOOGLE_API_KEY in the server .env."
    )


@app.post("/api/search")
async def search(request: SearchRequest):
    """Run agent and return the complete result."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    api_key = resolve_api_key(request.api_key)
    
    try:
        result = await run_agent(request.query, api_key)
        return result
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """Run agent with SSE streaming for real-time status updates."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    api_key = resolve_api_key(request.api_key)
    
    async def event_generator():
        try:
            async for event_data in run_agent_stream(request.query, api_key):
                yield {"data": event_data}
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_json = json.dumps({"type": "error", "message": str(e)})
            yield {"data": error_json}
    
    return EventSourceResponse(event_generator())


@app.get("/api/config")
async def get_config():
    """Return public configuration details (e.g. if server has a default API key)."""
    return {
        "has_default_api_key": bool(settings.google_api_key and settings.google_api_key.strip()),
        "model": settings.model_name
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": settings.model_name,
        "has_default_api_key": bool(settings.google_api_key and settings.google_api_key.strip())
    }


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
