from typing import List
from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    PROJECT_NAME: str = "Enterprise API 2026"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str 

    SECRET_KEY: str  
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  

    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    REDIS_HOST: str = 'redis'
    REDIS_PORT: int = 6379

    GROQ_API_KEY: str

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
        extra="ignore" 
    )

settings = Settings()


