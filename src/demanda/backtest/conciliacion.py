"""Conciliación con E2: el sistema actual del backtest, medido con el protocolo de la sección 5 del EDA.

Orígenes diarios desde 2023-07-01 hasta la última fecha − h, y la suma de T+1 a T+h, solo con los h días
observados. El pronóstico es la misma expresión que usa `SistemaActual` (`senal_previa`) por h.
"""

from datetime import date, timedelta
from typing import Any

import polars as pl

from demanda.backtest.metricas import wape
from demanda.datos.contratos import SERIE
from demanda.features.matriz import senal_previa, unidades

INICIO_E2 = date(2023, 7, 1)


def origenes_diarios(panel: pl.DataFrame, factor: float, horizonte: int) -> pl.DataFrame:
    """Un origen por serie y fecha posible, con `y` (la suma real), `yhat` y si la suma está completa."""
    futuro = [unidades().shift(-k).over(SERIE) for k in range(1, horizonte + 1)]
    ultima = panel["date"].max()
    return (
        panel.sort([*SERIE, "date"])
        .select(
            *SERIE,
            pl.col("date").alias("origen"),
            pl.sum_horizontal(futuro).alias("y"),
            pl.all_horizontal(x.is_not_null() for x in futuro).alias("completo"),
            (senal_previa(factor) * horizonte).alias("yhat"),
        )
        .filter(pl.col("origen").is_between(pl.lit(INICIO_E2), pl.lit(ultima) - timedelta(days=horizonte)))
    )


def _referencia(resumen: dict[str, Any] | None) -> tuple[float | None, int | None]:
    sistema = (resumen or {}).get("sistema_actual", {})
    return (
        sistema.get("wape_pct", {}).get("senal_T_menos_1_factor_ajustado"),
        sistema.get("origenes", {}).get("evaluados"),
    )


def conciliar(
    panel: pl.DataFrame, factor: float, horizonte: int, resumen: dict[str, Any] | None
) -> dict[str, Any]:
    """WAPE semanal del sistema actual con el protocolo de E2 y su diferencia con `resumen` (o null)."""
    posibles = origenes_diarios(panel, factor, horizonte)
    evaluados = posibles.filter(pl.col("completo") & pl.col("yhat").is_not_null())
    valor = evaluados.select(wape() * 100).item() if evaluados.height else None
    referencia, origenes_e2 = _referencia(resumen)
    return {
        "protocolo": (
            f"orígenes diarios desde {INICIO_E2} hasta la última fecha − {horizonte}; suma de T+1 a "
            f"T+{horizonte} completa; sistema actual = señal(T − 1) / factor × {horizonte}"
        ),
        "factor": factor,
        "origenes_posibles": posibles.height,
        "origenes_evaluados": evaluados.height,
        "origenes_evaluados_e2": origenes_e2,
        "wape_pct": None if valor is None else round(valor, 4),
        "referencia_e2_pct": referencia,
        "diferencia_pp": None if valor is None or referencia is None else round(valor - referencia, 4),
    }
