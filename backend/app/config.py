from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 127.0.0.1, not localhost: the port is only published on IPv4, and on Windows libpq
    # stalls on the IPv6 attempt until it times out.
    database_url: str = "postgresql://movie_app:movie_app_dev@127.0.0.1:5433/movie_app"
    # Read from ANTHROPIC_API_KEY. Without it the Claude conversation is switched off.
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-opus-5-5"
    # Chat needs quick turns. Raise it if the evaluation shows picks breaking the rules.
    claude_effort: str = "low"
    embedding_model_dir: str = str(Path(__file__).resolve().parents[2] / "data" / "models")

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_means_none(cls, value: object) -> object:
        # Docker Compose passes an unset key through as an empty string.
        return value or None


settings = Settings()
