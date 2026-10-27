"""El bucle por corte y modelo. Cada modelo recibe solo la historia y el futuro del corte."""

import time
from collections.abc import Callable, Mapping
from typing import NamedTuple

import polars as pl

from demanda.backtest.metricas import escala_mase
from demanda.backtest.protocolo import Protocolo, separar, ultima_fecha
from demanda.datos.contratos import SERIE
from demanda.features.matriz import unidades
from demanda.modelos.registro import INTERVALO, SALIDA, Modelo

COLUMNAS = ("modelo", "origen", *SERIE, "date", "h", "y", "yhat", *INTERVALO)


class Resultado(NamedTuple):
    pronosticos: pl.DataFrame  # COLUMNAS, una fila por modelo × origen × serie × día
    escalas: pl.DataFrame  # origen, serie y escala del MASE, medida en la historia de cada corte


def validar_salida(pred: pl.DataFrame, futuro: pl.DataFrame, nombre: str) -> pl.DataFrame:
    """La salida debe tener `SALIDA`, exactamente una fila por fila de `futuro` y valores finitos.

    Un pronóstico nulo sacaría ese punto de la evaluación de un solo modelo, y los modelos dejarían de
    compararse sobre los mismos puntos. El intervalo es opcional, pero si viene, viene completo.
    """
    faltan = [c for c in SALIDA if c not in pred.columns]
    if faltan:
        raise ValueError(f"{nombre}: faltan columnas en el pronóstico: {', '.join(faltan)}")
    llave = [*SERIE, "date", "h"]
    sobran = pred.join(futuro, on=llave, how="anti").height
    faltan_filas = futuro.join(pred, on=llave, how="anti").height
    if pred.height != futuro.height or sobran or faltan_filas:
        raise ValueError(f"{nombre}: el pronóstico no tiene exactamente una fila por fila de futuro")
    for c in ("yhat", *(c for c in INTERVALO if c in pred.columns)):
        if not pred.select(pl.col(c).cast(pl.Float64).is_finite().fill_null(False).all()).item():
            raise ValueError(f"{nombre}: `{c}` tiene valores nulos o no finitos")
    return pred.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(c) for c in INTERVALO if c not in pred.columns
    )


def correr(
    panel: pl.DataFrame,
    modelos: Mapping[str, Modelo],
    protocolo: Protocolo,
    progreso: Callable[[str], None] | None = None,
) -> Resultado:
    """Cada modelo en cada corte. El progreso reporta los segundos por modelo, que no van a las salidas."""
    pronosticos, escalas = [], []
    total = dict.fromkeys(modelos, 0.0)
    for origen in protocolo.origenes(ultima_fecha(panel)):
        corte = separar(panel, origen, protocolo.horizonte)
        escalas.append(escala_mase(corte.historia).with_columns(pl.lit(origen).alias("origen")))
        segundos = {}
        for nombre, modelo in modelos.items():
            inicio = time.perf_counter()
            pred = validar_salida(modelo.pronosticar(corte.historia, corte.futuro), corte.futuro, nombre)
            segundos[nombre] = time.perf_counter() - inicio
            total[nombre] += segundos[nombre]
            pronosticos.append(
                pred.select(
                    pl.lit(nombre).alias("modelo"),
                    pl.lit(origen).alias("origen"),
                    *SERIE,
                    "date",
                    pl.col("h").cast(pl.Int64),
                    *(pl.col(c).cast(pl.Float64) for c in ("yhat", *INTERVALO)),
                )
            )
        if progreso:
            progreso(f"corte {origen} listo · " + " · ".join(f"{n} {t:.1f} s" for n, t in segundos.items()))
    if progreso:
        progreso("tiempo por modelo: " + " · ".join(f"{n} {t:.1f} s" for n, t in total.items()))
    reales = panel.select(*SERIE, "date", unidades().alias("y"))
    return Resultado(
        pl.concat(pronosticos)
        .join(reales, on=[*SERIE, "date"], how="left", maintain_order="left")
        .select(COLUMNAS),
        pl.concat(escalas).select("origen", *SERIE, "escala"),
    )
