"""El newsvendor semanal como funciones puras: cuantil ponderado, ventana expansiva, políticas y métricas.

- Error semanal multiplicativo: r = Σy / Σŷ por serie y corte, solo en los pares completos. El nivel
  objetivo es S = Σŷ × q, con q el multiplicador de servicio de la categoría.
- q es el cuantil ponderado de r en la fracción crítica, con peso Σŷ. Con S = ŷᵢ · q, el costo del par i es
  ŷᵢ · ρ(rᵢ − q), así que ese cuantil minimiza el costo de la categoría en unidades. Los pares con Σŷ = 0 no
  entran a la calibración.
- Ventana expansiva: el multiplicador del corte k usa las semanas de calibración y los cortes j < k del
  backtest con evento `resto`, cuyos reales ya se conocen en T_k. Las semanas con evento no entran: los pares
  del Buen Fin, con r ≈ 2, empujarían el cuantil de todas las semanas siguientes.
"""

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from demanda.datos.contratos import SERIE
from demanda.decision.supuestos import Supuestos

NORMAL = "resto"
TOLERANCIA = 1e-12  # relativa, al comparar la suma acumulada de pesos con la fracción crítica


def pares_semanales(pronosticos: pl.DataFrame, horizonte: int) -> pl.DataFrame:
    """Por modelo × corte × serie, Σy y Σŷ, solo en los pares con los `horizonte` días observados."""
    return (
        pronosticos.group_by("modelo", "origen", *SERIE)
        .agg(
            pl.col("y").sum(),
            pl.col("yhat").sum(),
            ((pl.len() == horizonte) & pl.col("y").is_not_null().all()).alias("completo"),
        )
        .filter("completo")
        .drop("completo")
        .sort("modelo", "origen", *SERIE)
    )


def cuantil_ponderado(
    valores: Sequence[float] | np.ndarray, pesos: Sequence[float] | np.ndarray, fraccion: float
) -> float:
    """El menor valor cuya suma acumulada de pesos, en orden creciente, llega a `fraccion` del total. Sin
    interpolar; los pesos ≤ 0 no entran."""
    v, w = np.asarray(valores, dtype=float), np.asarray(pesos, dtype=float)
    entran = w > 0
    v, w = v[entran], w[entran]
    if not v.size:
        raise ValueError("no hay pares con peso positivo para calibrar el cuantil")
    orden = np.argsort(v, kind="stable")
    acumulado = np.cumsum(w[orden])
    k = int(np.searchsorted(acumulado, fraccion * acumulado[-1] * (1 - TOLERANCIA), side="left"))
    return float(v[orden][min(k, v.size - 1)])


def ventana(pares: pl.DataFrame, corte: date, horizonte: int) -> pl.DataFrame:
    """Los pares que puede usar el multiplicador del corte: de semanas normales (`normal`) y con el horizonte
    cerrado en T, así sus reales ya se conocen."""
    cerrado = pl.col("origen") + pl.duration(days=horizonte) <= pl.lit(corte)
    return pares.filter(pl.col("normal") & cerrado)


def multiplicadores(
    pares: pl.DataFrame, cortes: Sequence[date], fracciones: Mapping[str, float], horizonte: int
) -> pl.DataFrame:
    """Por corte × modelo × categoría, el multiplicador q y cuántos pares y semanas tiene su ventana.

    `pares` trae modelo, origen, serie, `y`, `yhat` y `normal`: las semanas de calibración y los cortes del
    backtest con evento `resto` van con `normal`.
    """
    modelos = sorted(pares["modelo"].unique().to_list())
    filas = []
    for corte in cortes:
        disponibles = ventana(pares, corte, horizonte).filter(pl.col("yhat") > 0)
        for modelo in modelos:
            for categoria, fc in fracciones.items():
                grupo = disponibles.filter((pl.col("modelo") == modelo) & (pl.col("category") == categoria))
                if grupo.is_empty():
                    raise ValueError(f"sin pares para calibrar {modelo} en {categoria} al corte {corte}")
                r = (grupo["y"] / grupo["yhat"]).to_numpy()
                filas.append(
                    {
                        "origen": corte,
                        "modelo": modelo,
                        "category": categoria,
                        "q": cuantil_ponderado(r, grupo["yhat"].to_numpy(), fc),
                        "pares": grupo.height,
                        "semanas": grupo["origen"].n_unique(),
                    }
                )
    return pl.DataFrame(
        filas,
        schema={
            "origen": pl.Date,
            "modelo": pl.String,
            "category": pl.String,
            "q": pl.Float64,
            "pares": pl.Int64,
            "semanas": pl.Int64,
        },
    )


def nombres(campeon: str, retador: str) -> dict[str, str]:
    """Las tres políticas, con su definición."""
    return {
        f"puntual_{campeon}": f"S = Σŷ de {campeon}, sin colchón: la referencia",
        f"newsvendor_{campeon}": f"S = Σŷ × q, con q calibrado sobre los errores de {campeon}",
        f"newsvendor_{retador}": f"S = Σŷ × q, con q calibrado sobre los errores de {retador}",
    }


def niveles_objetivo(pares: pl.DataFrame, mult: pl.DataFrame, campeon: str, retador: str) -> pl.DataFrame:
    """Por política × corte × serie, el real `y` y el nivel objetivo `S`, en los pares del backtest."""
    columnas = ["politica", "origen", *SERIE, "y", "S"]
    puntual = pares.filter(pl.col("modelo") == campeon).select(
        pl.lit(f"puntual_{campeon}").alias("politica"), "origen", *SERIE, "y", pl.col("yhat").alias("S")
    )
    con_colchon = (
        pares.filter(pl.col("modelo").is_in([campeon, retador]))
        .join(mult, on=["origen", "modelo", "category"], how="left", maintain_order="left")
        .with_columns(
            ("newsvendor_" + pl.col("modelo")).alias("politica"), (pl.col("yhat") * pl.col("q")).alias("S")
        )
    )
    if con_colchon["q"].null_count():
        raise ValueError("hay pares del backtest sin multiplicador")
    return pl.concat([puntual, con_colchon.select(columnas)])


def precio_medio(panel: pl.DataFrame, hasta: date) -> dict[str, float]:
    """Por categoría, Σ amount_total / Σ units_sold en la historia hasta `hasta`, en filas con unidades."""
    precios = (
        panel.filter((pl.col("date") <= hasta) & pl.col("units_sold").is_not_null())
        .group_by("category")
        .agg((pl.col("amount_total").sum() / pl.col("units_sold").sum()).alias("precio"))
        .sort("category")
    )
    return dict(precios.iter_rows())


def costos_unitarios(
    precios: Mapping[str, float], supuestos: Supuestos, fraccion: float | None = None
) -> pl.DataFrame:
    """Por categoría, Cu y Co en pesos por unidad y la fracción crítica.

    Con `fraccion` (la sensibilidad), todas las categorías usan esa fracción crítica: se mantiene Cu y
    Co = Cu · (1 − FC) / FC.
    """
    filas = []
    for categoria, costo in supuestos.costos.items():
        cu = costo.margen * precios[categoria]
        if fraccion is None:
            co, fc = costo.mantener * precios[categoria], costo.fraccion_critica
        else:
            co, fc = cu * (1 - fraccion) / fraccion, fraccion
        filas.append({"category": categoria, "cu": cu, "co": co, "fc": fc})
    return pl.DataFrame(filas)


def resultados(niveles: pl.DataFrame, costos: pl.DataFrame) -> pl.DataFrame:
    """Cada par con su sobreinventario max(S − y, 0), su quiebre max(y − S, 0) y su costo en pesos,
    Cu · quiebre + Co · sobreinventario."""
    return (
        niveles.join(costos, on="category", how="left", maintain_order="left")
        .sort("politica", "origen", *SERIE)
        .with_columns(
            (pl.col("S") - pl.col("y")).clip(lower_bound=0).alias("sobre"),
            (pl.col("y") - pl.col("S")).clip(lower_bound=0).alias("quiebre"),
        )
        .with_columns((pl.col("cu") * pl.col("quiebre") + pl.col("co") * pl.col("sobre")).alias("costo"))
    )


def metricas(res: pl.DataFrame, por: Sequence[str]) -> pl.DataFrame:
    """Por grupo: pares, demanda, stock, sobreinventario y quiebre en unidades y como % de la demanda, nivel
    de servicio de ciclo (% de pares sin quiebre) contra la fracción crítica media, fill rate, costo y stock
    sobre demanda."""
    return (
        res.group_by(por)
        .agg(
            pl.len().alias("pares"),
            pl.col("y").sum().alias("demanda"),
            pl.col("S").sum().alias("stock"),
            pl.col("sobre").sum().alias("sobreinventario"),
            pl.col("quiebre").sum().alias("quiebre"),
            ((pl.col("quiebre") == 0).mean() * 100).alias("nivel_servicio_pct"),
            (pl.col("fc").mean() * 100).alias("fc_objetivo_pct"),
            pl.col("costo").sum().alias("costo"),
        )
        .with_columns(
            (100 * pl.col("sobreinventario") / pl.col("demanda")).alias("sobreinventario_pct"),
            (100 * pl.col("quiebre") / pl.col("demanda")).alias("quiebre_pct"),
            (100 * (1 - pl.col("quiebre") / pl.col("demanda"))).alias("fill_rate_pct"),
            (pl.col("stock") / pl.col("demanda")).alias("stock_sobre_demanda"),
        )
    )


def costo_por_corte(res: pl.DataFrame) -> pl.DataFrame:
    """Por política y corte, el costo total en pesos: la serie de la prueba Diebold-Mariano."""
    return res.group_by("politica", "origen").agg(pl.col("costo").sum()).sort("politica", "origen")
