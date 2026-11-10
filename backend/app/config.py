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
    # Once Claude has cost this much in a UTC day, chat pauses until midnight UTC. Quick picks
    # keep working. Set a spending limit on the API key too, because turns already under way
    # can go past this.
    daily_spend_limit_usd: float = 5.0
    # Per visitor, counted by IP address in memory, so nothing about visitors is stored.
    chat_messages_per_hour: int = 30
    requests_per_minute: int = 120
    # Conversations older than this are deleted when the server starts.
    conversation_retention_days: int = 30
    # The public host names the app is served on, comma-separated, such as
    # "tonights-pick.example.com". The MCP endpoint rejects requests addressed to any other name.
    public_hosts: str = ""
    # The built frontend. When the folder exists, the API server also serves the app.
    static_dir: str = str(Path(__file__).resolve().parents[2] / "frontend" / "dist")

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_means_none(cls, value: object) -> object:
        # Docker Compose passes an unset key through as an empty string.
        return value or None


settings = Settings()
