"""Chronos-2 sin entrenar: el contexto completo de cada serie y las covariables futuras conocidas.

- `amazon/chronos-2` (Apache-2.0) en la revisión fijada, en CPU y sin ajuste fino.
- Contexto: toda la historia de cada serie, con los nulos como NaN; Chronos-2 maneja los faltantes.
- Covariables futuras: la promoción y las banderas del calendario, como 0/1. No usa la señal.
- `cross_learning=False`: cada serie se pronostica sola, y el resultado no depende de cómo se arman los lotes.

Los pesos se cargan una sola vez por instancia, con un cargador inyectable: el backtest crea los modelos una
vez, y las pruebas sustituyen el cargador por un doble que no descarga nada. El cargador real deja el
pipeline en un proceso aparte (ver `demanda.proceso_chronos`): torch y LightGBM no pueden compartir proceso.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from demanda.datos.contratos import SERIE
from demanda.features.matriz import unidades
from demanda.modelos.registro import ordenar_cuantiles, registrar
from demanda.proceso_chronos import MODELO, REVISION, PipelineAislado

if TYPE_CHECKING:
    import pandas as pd

COVARIABLES = (
    "has_promotion",
    "is_buen_fin",
    "is_navidad_season",
    "is_payday",
    "is_weekend",
    "is_holiday",
    "is_semana_santa",
)
CUANTILES = {0.1: "p10", 0.5: "yhat", 0.9: "p90"}
SEPARADOR = "|"


class Pipeline(Protocol):
    def predict_df(
        self, df: "pd.DataFrame", future_df: "pd.DataFrame | None" = None, **kwargs: Any
    ) -> Any: ...


Cargador = Callable[[], Pipeline]


def cargar_pipeline(solo_local: bool = False) -> Pipeline:
    """Los pesos de Hugging Face en la revisión fijada, en CPU y en un proceso aparte. La primera vez se
    descargan (~480 MB); con `solo_local`, solo se leen de la caché."""
    return PipelineAislado(solo_local)


# El que usa el registro. Se resuelve al crear el modelo, así las pruebas lo sustituyen por un doble.
CARGADOR: Cargador = cargar_pipeline


def _a_pandas(tabla: pl.DataFrame, con_objetivo: bool) -> "pd.DataFrame":
    """Formato largo de Chronos: `item_id`, `timestamp`, el objetivo (si va) y las covariables como 0/1."""
    objetivo = [unidades().alias("target")] if con_objetivo else []
    return (
        tabla.sort([*SERIE, "date"])
        .select(
            pl.concat_str(SERIE, separator=SEPARADOR).alias("item_id"),
            pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
            *objetivo,
            *(pl.col(c).cast(pl.Float64) for c in COVARIABLES),
        )
        .to_pandas()
    )


@dataclass(eq=False)
class Chronos2:
    """Chronos-2 sin entrenar, con el contexto completo y las covariables futuras conocidas."""

    cargador: Cargador = field(default=cargar_pipeline, repr=False)
    _pipeline: Pipeline | None = field(default=None, init=False, repr=False)

    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = self.cargador()
        return self._pipeline

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        crudo = self.pipeline().predict_df(
            _a_pandas(historia, con_objetivo=True),
            future_df=_a_pandas(futuro, con_objetivo=False),
            prediction_length=futuro.select(pl.col("h").max()).item(),
            quantile_levels=list(CUANTILES),
            freq="D",
            cross_learning=False,
        )
        pred = pl.from_pandas(crudo).select(
            pl.col("item_id").str.split_exact(SEPARADOR, 1).struct.rename_fields(list(SERIE)).struct.unnest(),
            pl.col("timestamp").cast(pl.Date).alias("date"),
            *(pl.col(str(tau)).cast(pl.Float64).alias(c) for tau, c in CUANTILES.items()),
        )
        llave = [*SERIE, "date", "h"]
        return ordenar_cuantiles(futuro.select(llave).join(pred, on=[*SERIE, "date"], how="left"))

    def cerrar(self) -> None:
        """Libera el pipeline (y su proceso, si tiene); la siguiente predicción lo vuelve a cargar."""
        cerrar = getattr(self._pipeline, "cerrar", None)
        if callable(cerrar):
            cerrar()
        self._pipeline = None

    def parametros(self) -> dict[str, Any]:
        """Lo que identifica la configuración, para MLflow."""
        return {"pesos": MODELO, "revision": REVISION, "covariables": list(COVARIABLES)}


registrar("chronos2", lambda ctx: Chronos2(cargador=CARGADOR))
