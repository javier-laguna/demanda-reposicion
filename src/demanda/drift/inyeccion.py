"""Drift inyectado sobre copias en memoria del panel, para validar el monitor y medir su potencia.

Cada escenario cambia solo los días T + 1 a T + horizonte del panel que recibe el monitor. Los pronósticos no
se tocan: se hicieron en T, antes del cambio.
- `falla_pos`: `units_sold` en nulo en todas las categorías de la primera tienda, del día 2 al 4 de la semana;
- `salto_nivel`: `units_sold` × 1.30 en todas las series de la región Centro, toda la semana;
- `cambio_promocion`: en Bebidas, `has_promotion` = 1 en filas sin promoción elegidas con semilla 0, hasta
  llegar a 40 %.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from demanda.datos.contratos import SERIE


def _en_semana(origen: date, horizonte: int) -> pl.Expr:
    return pl.col("date").is_between(origen + timedelta(days=1), origen + timedelta(days=horizonte))


def primera_tienda(panel: pl.DataFrame) -> str:
    return str(panel["store_id"].min())


def falla_pos(
    panel: pl.DataFrame, origen: date, dias: int = 3, desde: int = 2, tienda: str | None = None
) -> pl.DataFrame:
    """`units_sold` en nulo en todas las categorías de `tienda` (la primera, si no se da), de T + `desde` a
    T + `desde` + `dias` − 1. La bandera de nulo sigue al valor."""
    tienda = tienda or primera_tienda(panel)
    afectadas = (pl.col("store_id") == tienda) & pl.col("date").is_between(
        origen + timedelta(days=desde), origen + timedelta(days=desde + dias - 1)
    )
    return panel.with_columns(
        pl.when(afectadas).then(None).otherwise(pl.col("units_sold")).alias("units_sold"),
        (pl.col("flag_nulo_units_sold") | (afectadas & ~pl.col("flag_fila_faltante"))).alias(
            "flag_nulo_units_sold"
        ),
    )


def salto_nivel(
    panel: pl.DataFrame, origen: date, horizonte: int, factor: float, columna: str, nivel: str
) -> pl.DataFrame:
    """`units_sold` × `factor`, redondeado a unidades, en las series de `columna` = `nivel` en la semana."""
    afectadas = (pl.col(columna) == nivel) & _en_semana(origen, horizonte)
    saltado = (pl.col("units_sold").cast(pl.Float64) * factor).round().cast(pl.Int64)
    return panel.with_columns(
        pl.when(afectadas).then(saltado).otherwise(pl.col("units_sold")).alias("units_sold")
    )


def cambio_promocion(
    panel: pl.DataFrame,
    origen: date,
    horizonte: int,
    categoria: str = "Bebidas",
    objetivo: float = 0.40,
    semilla: int = 0,
) -> pl.DataFrame:
    """Pone `has_promotion` = 1 en filas sin promoción de `categoria` en la semana, elegidas con `semilla`,
    hasta que la tasa de la categoría llega a `objetivo`."""
    semana = (
        (pl.col("category") == categoria)
        & _en_semana(origen, horizonte)
        & pl.col("has_promotion").is_not_null()
    )
    filas = panel.with_row_index("_fila").filter(semana).sort(*SERIE, "date")
    faltan = math.ceil(objetivo * filas.height) - int(filas["has_promotion"].sum())
    candidatas = filas.filter(pl.col("has_promotion") == 0)["_fila"].to_numpy()
    if faltan <= 0:
        return panel
    elegidas = np.random.default_rng(semilla).choice(
        candidatas, size=min(faltan, candidatas.size), replace=False
    )
    marcar = pl.int_range(pl.len(), dtype=pl.UInt32).is_in(pl.Series(elegidas, dtype=pl.UInt32).implode())
    return panel.with_columns(
        pl.when(marcar).then(1).otherwise(pl.col("has_promotion")).alias("has_promotion")
    )


class Esperada(NamedTuple):
    """Una alerta que el escenario debe disparar: inesperada y con severidad de alerta."""

    chequeo: str
    segmento: str
    nivel: str
    por_modelo: bool = False  # una por cada modelo con pronóstico


class Escenario(NamedTuple):
    inyectar: Callable[[pl.DataFrame, date, int], pl.DataFrame]
    descripcion: str
    esperadas: Callable[[pl.DataFrame], tuple[Esperada, ...]]  # dependen del panel (la primera tienda)
    recomendacion: str


ESCENARIOS: Mapping[str, Escenario] = {
    "falla_pos": Escenario(
        lambda p, o, h: falla_pos(p, o),
        "units_sold en nulo en las 6 categorías de la primera tienda, del día 2 al 4 de la semana",
        lambda p: (Esperada("tienda_sin_datos", "tienda", primera_tienda(p)),),
        "revisar_captura",
    ),
    "salto_nivel": Escenario(
        lambda p, o, h: salto_nivel(p, o, h, 1.30, "region", "Centro"),
        "units_sold × 1.30 en todas las series de la región Centro, toda la semana",
        lambda p: (
            Esperada("nivel_demanda", "region", "Centro"),
            Esperada("sesgo_localizado", "region", "Centro", por_modelo=True),
        ),
        "reentrenar",
    ),
    "cambio_promocion": Escenario(
        lambda p, o, h: cambio_promocion(p, o, h),
        "en Bebidas, has_promotion = 1 en filas sin promoción elegidas con semilla 0, hasta 40 %",
        lambda p: (Esperada("plan_promocion", "categoria", "Bebidas"),),
        "reestimar_promocion",
    ),
}


def encontradas(
    alertas: Sequence[Mapping[str, Any]], esperada: Esperada, modelos: Sequence[str]
) -> dict[str | None, bool]:
    """Por modelo (o None si el chequeo no es por modelo), si la alerta esperada está en su segmento:
    inesperada y con severidad de alerta."""

    def hay(modelo: str | None) -> bool:
        return any(
            a["chequeo"] == esperada.chequeo
            and a["segmento"] == esperada.segmento
            and a["nivel"] == esperada.nivel
            and a["severidad"] == "alerta"
            and a["clasificacion"] == "inesperada"
            and (modelo is None or a["modelo"] == modelo)
            for a in alertas
        )

    return {m: hay(m) for m in modelos} if esperada.por_modelo else {None: hay(None)}
