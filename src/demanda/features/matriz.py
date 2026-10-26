"""Matriz para la estrategia directa: una fila por serie × origen T × horizonte h, con d = T + h el objetivo.

Lo que se calcula con `units_sold` o con la señal usa solo datos fechados en T o antes (la señal, en T − 1):
- las features del origen son ventanas que terminan en T, calculadas por serie sobre la malla completa y
  unidas por la fecha de origen; equivalen a desplazar h días antes de la ventana, contado desde d;
- los rezagos estacionales y(d − 7k) caen en T o antes porque h ≤ 7.

Las exógenas del día objetivo son las que se conocen de antemano: calendario, tienda y promoción. Los nulos se
quedan como nulos; nada se imputa.
"""

from collections.abc import Sequence
from datetime import date

import polars as pl

from demanda.datos.calidad import SENAL
from demanda.datos.contratos import SERIE, TIENDAS

REZAGOS = (7, 14, 21, 28)
DEL_ORIGEN = ("y_origen", "media_7", "media_28", "desv_28", "senal_previa")
REZAGOS_ESTACIONALES = tuple(f"y_d_menos_{k}" for k in REZAGOS)
CALENDARIO_OBJETIVO = (
    "day_of_week",
    "month",
    "is_payday",
    "holiday_name",
    "is_buen_fin",
    "is_navidad_season",
    "is_semana_santa",
    "is_weekend",
)
ATRIBUTOS_TIENDA = tuple(c for c in TIENDAS.columns if c != "store_id")
EXOGENAS = (*CALENDARIO_OBJETIVO, "has_promotion")
ESTATICAS = ("category", *ATRIBUTOS_TIENDA, "h")
FEATURES = (*DEL_ORIGEN, *REZAGOS_ESTACIONALES, *EXOGENAS, *ESTATICAS)
OBJETIVO = "y"


def unidades() -> pl.Expr:
    return pl.col("units_sold").cast(pl.Float64)


def senal_previa(factor: float) -> pl.Expr:
    """La señal del día anterior entre el factor: el pronóstico diario del sistema actual sin la fuga.

    Evaluada en la fila del origen T da `señal(T − 1) / factor`. Supone el panel ordenado por serie y fecha
    sobre la malla completa, para que el desplazamiento sea de un día de calendario.
    """
    return (pl.col(SENAL).shift(1) / factor).over(SERIE)


def _del_origen(panel: pl.DataFrame, factor: float) -> pl.DataFrame:
    y = unidades()
    return panel.sort([*SERIE, "date"]).select(
        *SERIE,
        pl.col("date").alias("origen"),
        y.alias("y_origen"),
        y.rolling_mean(7, min_samples=1).over(SERIE).alias("media_7"),
        y.rolling_mean(28, min_samples=1).over(SERIE).alias("media_28"),
        y.rolling_std(28, min_samples=2).over(SERIE).alias("desv_28"),
        senal_previa(factor).alias("senal_previa"),
    )


def construir(panel: pl.DataFrame, origenes: Sequence[date], horizonte: int, factor: float) -> pl.DataFrame:
    """Matriz con `origen`, la serie, `date` (= d), las `FEATURES` y el objetivo `y` = units_sold de d."""
    if horizonte > min(REZAGOS):
        raise ValueError(f"con h > {min(REZAGOS)} el rezago y(d − {min(REZAGOS)}) caería después del origen")
    reales = panel.select(*SERIE, "date", unidades().alias(OBJETIVO))
    filas = (
        pl.DataFrame({"origen": list(origenes)}, schema={"origen": pl.Date})
        .join(pl.DataFrame({"h": range(1, horizonte + 1)}, schema={"h": pl.Int64}), how="cross")
        .join(panel.select(SERIE).unique(), how="cross")
        .with_columns((pl.col("origen") + pl.duration(days=pl.col("h"))).alias("date"))
    )
    matriz = filas.join(_del_origen(panel, factor), on=[*SERIE, "origen"], how="left")
    for k in REZAGOS:
        rezago = reales.select(
            *SERIE,
            (pl.col("date") + pl.duration(days=k)).alias("date"),
            pl.col(OBJETIVO).alias(f"y_d_menos_{k}"),
        )
        matriz = matriz.join(rezago, on=[*SERIE, "date"], how="left")
    calendario = panel.select("date", *CALENDARIO_OBJETIVO).unique("date")
    tiendas = panel.select("store_id", *ATRIBUTOS_TIENDA).unique("store_id")
    return (
        matriz.join(calendario, on="date", how="left")
        .join(panel.select(*SERIE, "date", "has_promotion"), on=[*SERIE, "date"], how="left")
        .join(tiendas, on="store_id", how="left")
        .join(reales, on=[*SERIE, "date"], how="left")
        .select("origen", *SERIE, "date", *(c for c in FEATURES if c not in SERIE), OBJETIVO)
        .sort("origen", *SERIE, "h")
    )
