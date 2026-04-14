from typing import List
from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    PROJECT_NAME: str = "Enterprise API 2026"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URL: str | None = None

    @field_validator("SQLALCHEMY_DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info) -> str:
        if isinstance(v, str): return v
        
        return f"postgresql://{info.data['POSTGRES_USER']}:{info.data['POSTGRES_PASSWORD']}@{info.data['POSTGRES_SERVER']}/{info.data['POSTGRES_DB']}"

    
    SECRET_KEY: str  
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  

    
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    REDIS_HOST: str = 'redis'
    REDIS_PORT: int = "6379"

    GROQ_API_KEY: str

    
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True, 
        extra="ignore" 
    )

settings = Settings()


