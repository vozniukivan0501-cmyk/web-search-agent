"""
Entrypoint for running the AI Web Search Agent FastAPI application.
"""
import os
import uvicorn
from app.config import settings


if __name__ == "__main__":
    # Render and other cloud platforms provide the port via the PORT environment variable
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=port,
        reload=False
    )

