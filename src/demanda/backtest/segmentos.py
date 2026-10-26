"""Métricas por segmento en formato largo: granularidad, segmento, nivel, modelo, métrica, valor, n y n_total.

- Semanal: la suma de T+1 a T+h por serie y corte. Solo se evalúan los pares con los h días observados; `n`
  son esos pares y `n_total` todos, así la cobertura queda en la misma fila.
- Diaria: cada día con real y pronóstico; `n` son esos días.
"""

from collections.abc import Sequence
from typing import Any

import polars as pl

from demanda.backtest.bucle import Resultado
from demanda.backtest.metricas import cobertura, columna_cuantil, mase, pinball, sesgo, wape
from demanda.backtest.protocolo import Protocolo
from demanda.datos.contratos import SERIE

DIMENSIONES = {
    "formato": "store_format",
    "region": "region",
    "nse": "socioeconomic_level",
    "categoria": "category",
}
SEGMENTOS_SEMANALES = ("global", "evento", *DIMENSIONES, "corte")
SEGMENTOS_DIARIOS = (*SEGMENTOS_SEMANALES, "horizonte")
CLAVES = ("modelo", "segmento", "nivel")
PRIMERAS = ("wape_pct", "sesgo_pct", "mase")  # orden de las métricas en la tabla; luego pinball y cobertura


# Días que terminan en el origen: la ventana que promedian la señal y la MA7
VENTANA_POSTERIOR = 7


def evento(buen_fin: pl.Expr, navidad: pl.Expr, post_buen_fin: pl.Expr) -> pl.Expr:
    """Precedencia: Buen Fin, Navidad, la semana posterior al Buen Fin y el resto."""
    return (
        pl.when(buen_fin)
        .then(pl.lit("buen_fin"))
        .when(navidad)
        .then(pl.lit("navidad"))
        .when(post_buen_fin)
        .then(pl.lit("post_buen_fin"))
        .otherwise(pl.lit("resto"))
    )


def posterior_al_buen_fin(origenes: pl.Series, panel: pl.DataFrame) -> pl.DataFrame:
    """Por origen, si los días que terminan en T incluyen Buen Fin. Un pronóstico que promedia esa ventana
    arrastra el pico a la semana siguiente, así que esos cortes no cuentan como semanas normales."""
    dias = panel.filter(pl.col("is_buen_fin")).select(pl.col("date").unique().alias("dia"))
    desde_el_dia = (pl.col("origen") - pl.col("dia")).dt.total_days()
    return (
        pl.DataFrame({"origen": origenes.unique()})
        .join(dias, how="cross")
        .group_by("origen")
        .agg(desde_el_dia.is_between(0, VENTANA_POSTERIOR - 1).any().alias("post_buen_fin"))
    )


def tabla_diaria(resultado: Resultado, panel: pl.DataFrame) -> pl.DataFrame:
    """Los pronósticos con los niveles de cada segmento, la escala del MASE y si el día es evaluable."""
    atributos = panel.select(
        *SERIE,
        "date",
        *(c for c in DIMENSIONES.values() if c not in SERIE),
        "is_buen_fin",
        "is_navidad_season",
    )
    return (
        resultado.pronosticos.join(atributos, on=[*SERIE, "date"], how="left")
        .join(resultado.escalas, on=["origen", *SERIE], how="left")
        .join(posterior_al_buen_fin(resultado.pronosticos["origen"], panel), on="origen", how="left")
        .with_columns(pl.col("post_buen_fin").fill_null(False))
        .with_columns(
            pl.lit("global").alias("global"),
            evento(pl.col("is_buen_fin"), pl.col("is_navidad_season"), pl.col("post_buen_fin")).alias(
                "evento"
            ),
            *(pl.col(c).alias(nombre) for nombre, c in DIMENSIONES.items()),
            pl.col("origen").cast(pl.String).alias("corte"),
            pl.col("h").cast(pl.String).alias("horizonte"),
            pl.col("y").is_not_null().alias("evaluable"),
        )
    )


def tabla_semanal(diaria: pl.DataFrame, horizonte: int) -> pl.DataFrame:
    """Un par por modelo × serie × corte. Entra a Buen Fin o Navidad si algún día de su horizonte cae en el
    evento."""
    return diaria.group_by("modelo", "origen", *SERIE).agg(
        pl.col("y").sum(),
        pl.col("yhat").sum(),
        ((pl.len() == horizonte) & pl.col("evaluable").all()).alias("evaluable"),
        evento(
            pl.col("is_buen_fin").any(), pl.col("is_navidad_season").any(), pl.col("post_buen_fin").first()
        ).alias("evento"),
        *(pl.col(c).first() for c in ("global", *DIMENSIONES, "corte")),
    )


def metricas_semanales() -> list[pl.Expr]:
    return [(wape() * 100).alias("wape_pct"), (sesgo() * 100).alias("sesgo_pct")]


def metricas_diarias(cuantiles: Sequence[float]) -> list[pl.Expr]:
    """WAPE, sesgo, pinball por cuantil y, solo en modelos con intervalo, pinball medio y cobertura."""
    inferior, superior = columna_cuantil(min(cuantiles)), columna_cuantil(max(cuantiles))
    con_intervalo = pl.col(inferior).is_not_null().all() & pl.col(superior).is_not_null().all()
    perdidas = {tau: pinball(tau, columna_cuantil(tau)) * 100 for tau in cuantiles}
    por_cuantil = [
        (perdida if columna_cuantil(tau) == "yhat" else pl.when(con_intervalo).then(perdida)).alias(
            f"pinball_{round(tau * 100)}_pct"
        )
        for tau, perdida in perdidas.items()
    ]
    nominal = round((max(cuantiles) - min(cuantiles)) * 100)
    return [
        *metricas_semanales(),
        *por_cuantil,
        pl.when(con_intervalo).then(pl.mean_horizontal(list(perdidas.values()))).alias("pinball_medio_pct"),
        pl.when(con_intervalo).then(cobertura(inferior, superior) * 100).alias(f"cobertura_{nominal}_pct"),
    ]


def por_segmento(
    df: pl.DataFrame, segmentos: Sequence[str], metricas: list[pl.Expr], con_mase: bool
) -> pl.DataFrame:
    """Una fila por modelo × segmento × nivel, con `n`, `n_total` y una columna por métrica."""
    claves = list(CLAVES)
    partes = []
    for segmento in segmentos:
        d = df.with_columns(pl.lit(segmento).alias("segmento"), pl.col(segmento).alias("nivel"))
        evaluables = d.filter("evaluable")
        valores = evaluables.group_by(claves).agg(*metricas)
        if con_mase:
            valores = valores.join(mase(evaluables, claves), on=claves, how="left")
        conteos = d.group_by(claves).agg(
            pl.col("evaluable").sum().cast(pl.Int64).alias("n"), pl.len().cast(pl.Int64).alias("n_total")
        )
        partes.append(conteos.join(valores, on=claves, how="left"))
    return pl.concat(partes)


def _largo(ancho: pl.DataFrame, granularidad: str) -> pl.DataFrame:
    return (
        ancho.unpivot(index=[*CLAVES, "n", "n_total"], variable_name="metrica", value_name="valor")
        .drop_nulls("valor")
        .select(
            pl.lit(granularidad).alias("granularidad"),
            "segmento",
            "nivel",
            "modelo",
            "metrica",
            pl.col("valor").round(4),
            "n",
            "n_total",
        )
    )


def tabla(
    resultado: Resultado, panel: pl.DataFrame, protocolo: Protocolo, modelos: Sequence[str]
) -> pl.DataFrame:
    """Todas las métricas por segmento, semanales y diarias, en formato largo y en orden estable."""
    diaria = tabla_diaria(resultado, panel)
    semanal = tabla_semanal(diaria, protocolo.horizonte)
    ancho_diario = por_segmento(
        diaria, SEGMENTOS_DIARIOS, metricas_diarias(protocolo.cuantiles), con_mase=True
    )
    largo = pl.concat(
        [
            _largo(
                por_segmento(semanal, SEGMENTOS_SEMANALES, metricas_semanales(), con_mase=False), "semanal"
            ),
            _largo(ancho_diario, "diaria"),
        ]
    )
    resto = [c for c in ancho_diario.columns if c not in (*CLAVES, "n", "n_total", *PRIMERAS)]
    orden = {
        "granularidad": ["semanal", "diaria"],
        "segmento": list(SEGMENTOS_DIARIOS),
        "modelo": list(modelos),
        "metrica": [*PRIMERAS, *resto],
    }
    return (
        largo.with_columns(pl.col(c).cast(pl.Enum(v)) for c, v in orden.items())
        .sort("granularidad", "segmento", "nivel", "modelo", "metrica")
        .with_columns(pl.col(c).cast(pl.String) for c in orden)
    )


def globales(tabla: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Por modelo, las métricas del segmento global como `<granularidad>_<métrica>`, más `n` y `n_total`."""
    resumen: dict[str, dict[str, Any]] = {}
    for fila in tabla.filter(pl.col("segmento") == "global").iter_rows(named=True):
        g = fila["granularidad"]
        resumen.setdefault(fila["modelo"], {}).update(
            {f"{g}_{fila['metrica']}": fila["valor"], f"{g}_n": fila["n"], f"{g}_n_total": fila["n_total"]}
        )
    return resumen
