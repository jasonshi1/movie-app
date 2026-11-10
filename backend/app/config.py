from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 127.0.0.1, not localhost: the port is only published on IPv4, and on Windows libpq
    # stalls on the IPv6 attempt until it times out.
    database_url: str = "postgresql://movie_app:movie_app_dev@127.0.0.1:5433/movie_app"


settings = Settings()
