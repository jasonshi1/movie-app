from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://movie_app:movie_app_dev@localhost:5433/movie_app"


settings = Settings()
