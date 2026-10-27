"""Registro en MLflow: un run por modelo con el protocolo como parámetros, las métricas globales y
`metricas.json` como artefacto.

MLflow se importa dentro de la función: tarda en cargarse y, al importarse, imprime una sugerencia dirigida a
agentes de código que se silencia con `MLFLOW_DISABLE_AGENT_HINT`.
"""

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPERIMENTO = "backtest"


def registrar_runs(
    tracking_uri: str,
    mlruns_dir: Path,
    parametros: Mapping[str, Any],
    globales: Mapping[str, Mapping[str, Any]],
    metricas_path: Path,
    por_run: Mapping[str, Mapping[str, Any]] | None = None,
    experimento: str = EXPERIMENTO,
) -> list[str]:
    """Crea un run por modelo de `globales` y devuelve sus ids.

    `parametros` son los comunes; `por_run`, los propios de cada modelo o configuración.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    cliente = MlflowClient(tracking_uri=tracking_uri)
    existente = cliente.get_experiment_by_name(experimento)
    if existente is None:
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        id_experimento = cliente.create_experiment(
            experimento, artifact_location=mlruns_dir.resolve().as_uri()
        )
    else:
        id_experimento = existente.experiment_id

    ahora = int(time.time() * 1000)
    ids = []
    for modelo, valores in globales.items():
        run = cliente.create_run(id_experimento, run_name=modelo, tags={"modelo": modelo})
        cliente.log_batch(
            run.info.run_id,
            metrics=[Metric(k, float(v), ahora, 0) for k, v in valores.items() if v is not None],
            params=[
                Param(k, str(v))
                for k, v in {**parametros, **(por_run or {}).get(modelo, {}), "modelo": modelo}.items()
            ],
        )
        cliente.log_artifact(run.info.run_id, str(metricas_path))
        cliente.set_terminated(run.info.run_id)
        ids.append(run.info.run_id)
    return ids
