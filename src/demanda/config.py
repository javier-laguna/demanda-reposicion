"""Configuración del proyecto: rutas y proveedor LLM, leídos de variables de entorno o de `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = RAIZ / "data"
    artifacts_dir: Path = RAIZ / "artifacts"
    reports_dir: Path = RAIZ / "reports"

    # Capa de agentes. `llm_model` usa la notación de Pydantic AI (`proveedor:modelo`). Sin key se usa la
    # caché de insights o, si no coincide, las plantillas.
    llm_model: str = "anthropic:claude-sonnet-5-5"
    llm_base_url: str | None = None
    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"


@lru_cache
def get_config() -> Config:
    return Config()
