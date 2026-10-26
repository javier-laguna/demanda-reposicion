"""Protocolo del backtest: los orígenes y la separación, por construcción, entre historia y futuro."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import NamedTuple

import polars as pl

from demanda.datos.calidad import SENAL
from demanda.datos.contratos import CALENDARIO, LLAVE, SERIE, TIENDAS

# Columnas de la señal: la de T ya incluye las ventas de T + 1, así que en T no se conoce ninguna de las dos
DE_LA_SENAL = (SENAL, f"flag_nulo_{SENAL}")

# Lo que se sabe de antemano de los días por pronosticar. La promoción se supone conocida porque es un plan
# comercial; es un supuesto y queda documentado en `metricas.json`.
CONOCIDAS = (
    *LLAVE,
    *(c for c in TIENDAS.columns if c != "store_id"),
    *(c for c in CALENDARIO.columns if c != "date"),
    "has_promotion",
)


@dataclass(frozen=True)
class Protocolo:
    n_cortes: int = 17
    horizonte: int = 7
    cuantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def origenes(self, ultima: date) -> list[date]:
        """T = última − horizonte · k, k = 1..n_cortes, en orden cronológico.

        Los horizontes no se traslapan y el del último corte termina en la última fecha.
        """
        return [ultima - timedelta(days=self.horizonte * k) for k in range(self.n_cortes, 0, -1)]


def ultima_fecha(panel: pl.DataFrame) -> date:
    ultima = panel["date"].max()
    if not isinstance(ultima, date):
        raise ValueError("el panel no tiene fechas")
    return ultima


class Corte(NamedTuple):
    origen: date
    historia: pl.DataFrame
    futuro: pl.DataFrame


def separar(panel: pl.DataFrame, origen: date, horizonte: int) -> Corte:
    """`historia`: el panel hasta T, con la señal de T y su bandera en nulo (ya incluye las ventas de T + 1).
    `futuro`: de T + 1 a T + horizonte, solo las columnas `CONOCIDAS` y `h`."""
    historia = (
        panel.filter(pl.col("date") <= origen)
        .with_columns(pl.when(pl.col("date") < origen).then(pl.col(c)).alias(c) for c in DE_LA_SENAL)
        .sort([*SERIE, "date"])
    )
    futuro = (
        panel.filter(
            pl.col("date").is_between(origen + timedelta(days=1), origen + timedelta(days=horizonte))
        )
        .select(*CONOCIDAS, (pl.col("date") - pl.lit(origen)).dt.total_days().cast(pl.Int64).alias("h"))
        .sort([*SERIE, "date"])
    )
    return Corte(origen, historia, futuro)
