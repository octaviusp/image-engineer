from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GOOGLE_FAST_MODEL: str = "gemini-2.0-flash"
    GOOGLE_PRO_MODEL: str = "gemini-2.5-pro-exp-03-25"
    GOOGLE_IMAGE_GENERATION_MODEL: str = "gemini-2.0-flash-exp-image-generation"
    GOOGLE_VIDEO_GENERATION_MODEL: str = "veo-2.0-generate-001"
    
    TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 2048
    MAX_CONCURRENCE_CALLS: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()
        
        
    
