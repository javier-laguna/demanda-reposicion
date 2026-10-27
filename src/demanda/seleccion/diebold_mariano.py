"""Diebold-Mariano sobre la pérdida por corte, con la corrección de Harvey, Leybourne y Newbold.

- Pérdida por corte: d_t = Σ_i |E_A,i,t| − Σ_i |E_B,i,t|, con E el error de la suma semanal y la suma sobre
  los pares completos del corte t. Sin normalizar por corte, la media de d_t es proporcional a la diferencia
  de WAPE semanal global: la prueba contrasta la métrica de decisión.
- Por corte y no por par: los errores de todas las series en un mismo corte comparten el choque de la semana.
  Tratarlos como independientes inflaría la significancia.
- Los horizontes semanales no se traslapan, así que la varianza HAC no lleva rezagos (h = 1 a frecuencia
  semanal). El estadístico corregido se compara contra una t con n − 1 grados de libertad, en dos colas.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from demanda.datos.contratos import SERIE

MIN_CORTES = 3


class PruebaDM(NamedTuple):
    n: int  # cortes
    media: float  # media de d_t: negativa si A se equivoca menos que B
    estadistico: float  # DM con la corrección HLN
    p_valor: float  # dos colas, t con n − 1 grados de libertad


def errores_semanales(pronosticos: pl.DataFrame, horizonte: int) -> pl.DataFrame:
    """Por modelo × corte × serie, la suma semanal del real y el error Σŷ − Σy, solo en los pares completos.

    Un par es completo si tiene los `horizonte` días con real, igual que en las métricas semanales.
    """
    return (
        pronosticos.group_by("modelo", "origen", *SERIE)
        .agg(
            pl.col("y").sum(),
            (pl.col("yhat").sum() - pl.col("y").sum()).alias("error"),
            ((pl.len() == horizonte) & pl.col("y").is_not_null().all()).alias("completo"),
        )
        .filter("completo")
        .drop("completo")
    )


def perdida_por_corte(errores: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Por `origen`, d_t = Σ|E_A| − Σ|E_B| y el real `y` = Σy, sobre los pares completos de los dos modelos.

    100 · Σd / Σy es la diferencia de WAPE semanal entre A y B en esos pares.
    """
    llave = ["origen", *SERIE]

    def de(modelo: str, *otras: str) -> pl.DataFrame:
        del_modelo = errores.filter(pl.col("modelo") == modelo)
        return del_modelo.select(*llave, *otras, pl.col("error").abs().alias(f"abs_{modelo}"))

    return (
        de(a, "y")
        .join(de(b), on=llave, how="inner")
        .group_by("origen")
        .agg((pl.col(f"abs_{a}").sum() - pl.col(f"abs_{b}").sum()).alias("d"), pl.col("y").sum())
        .sort("origen")
    )


def diebold_mariano(d: Sequence[float]) -> PruebaDM:
    """La prueba sobre la serie de pérdidas d_t, a un paso (h = 1) y con la corrección HLN.

    Sin rezagos, la varianza de la media es γ₀ / n, con γ₀ = Σ(d_t − d̄)² / n, y el factor de HLN es
    √((n + 1 − 2h + h(h − 1)/n) / n) = √((n − 1) / n). El estadístico corregido coincide con la t pareada.
    """
    x = np.asarray(d, dtype=float)
    n = x.size
    if n < MIN_CORTES:
        raise ValueError(f"Diebold-Mariano necesita al menos {MIN_CORTES} cortes; hay {n}")
    media = float(x.mean())
    gamma0 = float(((x - media) ** 2).sum()) / n
    if gamma0 == 0:
        raise ValueError("las pérdidas no varían entre cortes: la prueba no está definida")
    estadistico = math.sqrt((n - 1) / n) * media / math.sqrt(gamma0 / n)
    return PruebaDM(n, media, estadistico, float(2 * stats.t.sf(abs(estadistico), df=n - 1)))
