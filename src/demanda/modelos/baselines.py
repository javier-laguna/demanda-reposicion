"""Baselines sin calendario: el sistema actual, el naive estacional y la media de 7 días."""

from dataclasses import dataclass
from datetime import timedelta

import polars as pl

from demanda.datos.contratos import SERIE
from demanda.features.matriz import senal_previa, unidades
from demanda.modelos.registro import registrar

LLAVE_FUTURO = (*SERIE, "date", "h")


def _origen(historia: pl.DataFrame) -> pl.Expr:
    return pl.lit(historia["date"].max(), dtype=pl.Date)


def media_7(historia: pl.DataFrame) -> pl.DataFrame:
    """Media de los valores observados de T − 6 a T, por serie; nula si no hay ninguno."""
    return (
        historia.filter(pl.col("date") > _origen(historia) - pl.duration(days=7))
        .group_by(SERIE)
        .agg(unidades().mean().alias("ma7"))
    )


@dataclass(frozen=True)
class SistemaActual:
    """La señal fechada en T − 1 entre el factor, igual para todo h."""

    factor: float

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        en_origen = (
            historia.sort([*SERIE, "date"])
            .with_columns(senal_previa(self.factor).alias("yhat"))
            .filter(pl.col("date") == _origen(historia))
            .select(*SERIE, "yhat")
        )
        return futuro.select(LLAVE_FUTURO).join(en_origen, on=SERIE, how="left")


@dataclass(frozen=True)
class NaiveEstacional:
    """y(T + h − 7); si falta, y(T + h − 14); si también falta, la media de 7 días."""

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        y = historia.select(*SERIE, "date", unidades().alias("y"))
        pred = futuro.select(LLAVE_FUTURO)
        for k in (7, 14):
            hace_k = y.select(*SERIE, pl.col("date") + timedelta(days=k), pl.col("y").alias(f"y_{k}"))
            pred = pred.join(hace_k, on=[*SERIE, "date"], how="left")
        return pred.join(media_7(historia), on=SERIE, how="left").select(
            *LLAVE_FUTURO, pl.coalesce("y_7", "y_14", "ma7").alias("yhat")
        )


@dataclass(frozen=True)
class MediaMovil7:
    """La media de los valores observados de T − 6 a T, igual para todo h."""

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        return (
            futuro.select(LLAVE_FUTURO)
            .join(media_7(historia), on=SERIE, how="left")
            .select(*LLAVE_FUTURO, pl.col("ma7").alias("yhat"))
        )


registrar("sistema_actual", lambda ctx: SistemaActual(ctx.factor_senal))
registrar("naive_estacional", lambda ctx: NaiveEstacional())
registrar("ma7", lambda ctx: MediaMovil7())
