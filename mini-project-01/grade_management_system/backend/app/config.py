from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Grade Management System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database (SQLite for local development)
    DATABASE_URL: str = "sqlite:///./grade_db.sqlite"

    # JWT Settings
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS - can be "*" or comma-separated list
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS into a list"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


settings = Settings()
