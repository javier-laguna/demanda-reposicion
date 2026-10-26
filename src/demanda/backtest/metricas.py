"""Métricas como funciones puras sobre una tabla de pronósticos con reales.

`y` es el real y `yhat` el pronóstico; los cuantiles van en `p10`, `p90`, etc., y el de 0.5 es `yhat`. Las
expresiones agregan sobre las filas que reciben: las que no tienen real o pronóstico se filtran antes, porque
aquí no se imputa nada. WAPE, sesgo y pinball suman los errores del segmento antes de dividir entre Σy.
"""

from collections.abc import Sequence

import polars as pl

from demanda.datos.contratos import SERIE


def columna_cuantil(tau: float) -> str:
    return "yhat" if tau == 0.5 else f"p{round(tau * 100)}"


def wape(y: str = "y", yhat: str = "yhat") -> pl.Expr:
    """Σ|y − ŷ| / Σy."""
    return (pl.col(yhat) - pl.col(y)).abs().sum() / pl.col(y).sum()


def sesgo(y: str = "y", yhat: str = "yhat") -> pl.Expr:
    """Σ(ŷ − y) / Σy; positivo es sobrepronóstico."""
    return (pl.col(yhat) - pl.col(y)).sum() / pl.col(y).sum()


def pinball(tau: float, q: str, y: str = "y") -> pl.Expr:
    """Σ ρ_τ(y − q) / Σy, con ρ_τ(u) = max(τu, (τ − 1)u)."""
    u = pl.col(y) - pl.col(q)
    return pl.max_horizontal(tau * u, (tau - 1) * u).sum() / pl.col(y).sum()


def cobertura(inferior: str, superior: str, y: str = "y") -> pl.Expr:
    """Proporción de reales dentro de [inferior, superior]."""
    return pl.col(y).is_between(pl.col(inferior), pl.col(superior)).mean()


def escala_mase(historia: pl.DataFrame, temporada: int = 7) -> pl.DataFrame:
    """Por serie: el MAE del naive estacional dentro de la historia, media de |y(t) − y(t − temporada)|."""
    y = pl.col("units_sold").cast(pl.Float64)
    return (
        historia.sort([*SERIE, "date"])
        .with_columns((y - y.shift(temporada)).abs().over(SERIE).alias("error_naive"))
        .group_by(SERIE)
        .agg(pl.col("error_naive").mean().alias("escala"))
    )


def mase(df: pl.DataFrame, por: Sequence[str]) -> pl.DataFrame:
    """Por serie y corte, el MAE del horizonte entre `escala`; luego el promedio por grupo de `por`.

    Los pares sin escala positiva (una serie constante en su historia) no entran al promedio.
    """
    return (
        df.group_by(*por, "origen", *SERIE)
        .agg(((pl.col("yhat") - pl.col("y")).abs().mean() / pl.col("escala").first()).alias("mase"))
        .filter(pl.col("mase").is_finite())
        .group_by(por)
        .agg(pl.col("mase").mean())
    )
