"""Configuración del proyecto: rutas, MLflow y proveedor LLM, leídos de variables de entorno o de `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parents[2]


def _mlflow_por_defecto(campos: dict[str, Any]) -> str:
    return f"sqlite:///{Path(campos['artifacts_dir']).resolve() / 'mlflow.db'}"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = RAIZ / "data"
    artifacts_dir: Path = RAIZ / "artifacts"
    reports_dir: Path = RAIZ / "reports"
    # Por defecto vive junto a los artefactos, así nada del backtest se escribe fuera de `artifacts_dir`.
    # Se lee de una variable propia: la estándar, `MLFLOW_TRACKING_URI`, puede apuntar a un servidor
    # remoto en el shell
    mlflow_tracking_uri: str = Field(
        default_factory=_mlflow_por_defecto, validation_alias="DEMANDA_MLFLOW_TRACKING_URI"
    )

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

    @property
    def mlruns_dir(self) -> Path:
        """Donde MLflow guarda los artefactos de cada run."""
        return self.artifacts_dir / "mlruns"


@lru_cache
def get_config() -> Config:
    return Config()
