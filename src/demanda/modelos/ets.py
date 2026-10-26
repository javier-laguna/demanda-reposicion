"""AutoETS de statsforecast, por serie y reajustado en cada corte."""

from dataclasses import dataclass

import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from demanda.datos.contratos import SERIE
from demanda.features.matriz import unidades
from demanda.modelos.registro import registrar

NIVEL = 80  # intervalo central: sus extremos son p10 y p90
SEPARADOR = "|"


def rellenar_para_ajuste(historia: pl.DataFrame) -> pl.DataFrame:
    """`unique_id, ds, y` con los nulos rellenos, solo para ajustar: primero con el valor de una semana antes,
    después con el más cercano. Supone la malla completa ordenada por serie y fecha."""
    y = unidades()
    relleno = y.fill_null(y.shift(7)).interpolate(method="nearest").forward_fill().backward_fill()
    return (
        historia.sort([*SERIE, "date"])
        .select(
            pl.concat_str(SERIE, separator=SEPARADOR).alias("unique_id"),
            pl.col("date").alias("ds"),
            relleno.over(SERIE).alias("y"),
        )
        .drop_nulls("y")
    )


@dataclass(frozen=True)
class AutoETSModelo:
    """AutoETS con temporada 7 por serie, reajustado en cada corte; p10 y p90 de su intervalo del 80 %."""

    n_jobs: int = 1
    temporada: int = 7

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        sf = StatsForecast(models=[AutoETS(season_length=self.temporada)], freq="1d", n_jobs=self.n_jobs)
        crudo = sf.forecast(
            df=rellenar_para_ajuste(historia), h=futuro.select(pl.col("h").max()).item(), level=[NIVEL]
        )
        if not isinstance(crudo, pl.DataFrame):
            raise TypeError("statsforecast devolvió un DataFrame que no es de Polars")
        # Las unidades no son negativas: el intervalo se recorta en cero sin romper p10 ≤ yhat ≤ p90
        pred = crudo.select(
            pl.col("unique_id")
            .str.split_exact(SEPARADOR, 1)
            .struct.rename_fields(list(SERIE))
            .struct.unnest(),
            pl.col("ds").alias("date"),
            pl.col("AutoETS").clip(lower_bound=0).alias("yhat"),
            pl.col(f"AutoETS-lo-{NIVEL}").clip(lower_bound=0).alias("p10"),
            pl.col(f"AutoETS-hi-{NIVEL}").clip(lower_bound=0).alias("p90"),
        )
        return futuro.select(*SERIE, "date", "h").join(pred, on=[*SERIE, "date"], how="left")


registrar("autoets", lambda ctx: AutoETSModelo(n_jobs=ctx.n_jobs))
