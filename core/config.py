from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "pNInz6obpgDQGcFmaJgB" # Default to Adam
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
