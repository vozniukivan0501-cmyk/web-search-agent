from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: Optional[str] = None
    model_name: str = "models/gemini-3.6-flash"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    max_search_results: int = 10
    top_k_reranked: int = 3
    max_reasoning_steps: int = 5
    
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
