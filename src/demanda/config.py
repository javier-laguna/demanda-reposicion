"""Configuración del proyecto: rutas, MLflow y proveedor LLM, leídos de variables de entorno o de `.env`.

`DEMANDA_ENV_FILE` cambia el archivo `.env` que se lee (por omisión, el de la raíz del repo). Las pruebas
la apuntan a una ruta inexistente, así nunca leen las keys reales.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import ENV_FILE_SENTINEL

RAIZ = Path(__file__).resolve().parents[2]


def env_file() -> Path:
    """El `.env` que lee la config: `DEMANDA_ENV_FILE` si está definida, si no `RAIZ/.env`."""
    return Path(os.environ.get("DEMANDA_ENV_FILE") or RAIZ / ".env")


def _mlflow_por_defecto(campos: dict[str, Any]) -> str:
    return f"sqlite:///{Path(campos['artifacts_dir']).resolve() / 'mlflow.db'}"


def _mlruns_por_defecto(campos: dict[str, Any]) -> Path:
    return Path(campos["artifacts_dir"]) / "mlruns"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = RAIZ / "data"
    artifacts_dir: Path = RAIZ / "artifacts"
    reports_dir: Path = RAIZ / "reports"
    # Los diagramas de archify; el tablero muestra el de la carpeta más reciente
    archify_dir: Path = RAIZ / ".archify"
    # Por defecto vive junto a los artefactos, así nada del backtest se escribe fuera de `artifacts_dir`.
    # Se lee de una variable propia: la estándar, `MLFLOW_TRACKING_URI`, puede apuntar a un servidor
    # remoto en el shell
    mlflow_tracking_uri: str = Field(
        default_factory=_mlflow_por_defecto, validation_alias="DEMANDA_MLFLOW_TRACKING_URI"
    )
    # Donde MLflow guarda los artefactos de cada run. El experimento fija la ruta absoluta al crearse, así
    # que cada entorno (la laptop, Docker) usa su propia base y su propio directorio (ver compose.yaml)
    mlruns_dir: Path = Field(default_factory=_mlruns_por_defecto, validation_alias="DEMANDA_MLRUNS_DIR")
    # La caché de insights que se versiona (`cache.json`). La prueba de humo la apunta a un temporal, así
    # nunca lee ni escribe la del repo
    insights_dir: Path = RAIZ / "insights"

    # Capa de agentes. `llm_model` usa la notación de Pydantic AI (`proveedor:modelo`). Sin key se usa la
    # caché de insights o, si no coincide, las plantillas. `llm_base_url` es para el proveedor `openai`
    # (un gateway compatible con su API).
    llm_model: str = "anthropic:claude-opus-5-5"
    llm_base_url: str | None = None
    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Freno de gasto propio (el límite duro va en la consola del proveedor): el modo vivo no arranca si el
    # gasto acumulado de la máquina llega a `llm_presupuesto_usd`, y una corrida se corta al pasar
    # `llm_tope_corrida_usd`. Ambos en USD estimados con la tabla de precios de `agentes/proveedor.py`.
    llm_presupuesto_usd: float = 20.0
    llm_tope_corrida_usd: float = 1.0

    def __init__(self, _env_file: Any = ENV_FILE_SENTINEL, **valores: Any) -> None:
        # La ruta se resuelve al crear la config, no al importar el módulo; un `_env_file` explícito gana
        if _env_file is ENV_FILE_SENTINEL:
            _env_file = env_file()
        super().__init__(_env_file=_env_file, **valores)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"


@lru_cache
def get_config() -> Config:
    return Config()
