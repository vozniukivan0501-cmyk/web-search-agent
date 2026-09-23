"""
Описание: Точка входа для запуска AI Web Search Agent.
Input: None
Args: None
Output: None
"""
import uvicorn
from app.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
