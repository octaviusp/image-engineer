from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GOOGLE_FAST_MODEL: str = "gemini-2.0-flash-001"
    GOOGLE_MODEL: str = "gemini-2.0-flash-lite"
    GOOGLE_IMAGE_GENERATION_MODEL: str = "gemini-2.0-flash-exp-image-generation"
    GOOGLE_VIDEO_GENERATION_MODEL: str = "veo-2.0-generate-001"
    ELEVEN_LABS_API_KEY: str
    TEMPERATURE: float = 1
    MAX_TOKENS: int = 4096
    MAX_CONCURRENCE_CALLS: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()
        
        
    
