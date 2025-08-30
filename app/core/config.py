# app/core/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    env: str = "dev"
    database_url: str
    supabase_project_url: str
    supabase_jwks_url: str
    supabase_jwt_secret: str
    openai_api_key: str

    # Busca primero en variables de entorno y luego en .env
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()