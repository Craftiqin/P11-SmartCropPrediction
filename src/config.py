from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_base_url: str = "https://api.craftiq.in"

    @field_validator("api_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("API_BASE_URL must be an HTTP(S) origin without path or credentials")
        return value

