"""¿La asignación de la promoción es aleatoria? Las pruebas B1 a B5 y la corrección de Holm.

Solo usan variables que la promoción no puede mover. Cada una corre sobre una muestra:
- B1, B2 y B3, sobre la del efecto (filas con unidades, monto, transacciones y promoción): el balance que
  importa es el de las filas que entran a la estimación.
- B4 y B5, sobre todas las filas con la promoción observada: describen el proceso de asignación, y la
  atrición necesita las filas sin unidades.

Las χ² son de Pearson, sin corrección de continuidad. Las Wald conjuntas de B1 y B3 llevan la corrección de
Hotelling de `absorcion.wald`: la F(q, G − 1) sin corregir rechaza de más con 36 restricciones y 80 tiendas, y
va como `p_sin_correccion`. La familia de Holm son las 13 de B2, las 2 de B4 y las
2 de B5. La aleatorización se sostiene si B1 y B3 dan p ≥ 0.01 y ninguna de la familia tiene p de Holm < 0.05.

Esa es la regla pre-registrada y se reporta tal cual. Aparte va si se sostienen las pruebas de la
asignación, sin las de `MEDICION`: `flag_conteo_inconsistente` se mide el mismo día que la promoción, así
que es un resultado y no una covariable. Esa separación se decidió después de ver que la bandera marca justo
los días con promoción.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.multitest import multipletests

from demanda.causal.absorcion import estimar, wald
from demanda.datos.contratos import SERIE

PROMO = "has_promotion"
ALFA_HOLM = 0.05
ALFA_CONJUNTA = 0.01  # B1 y B3
# Pruebas de la familia que miden el registro del mismo día y no la asignación (se separaron tras ver datos)
MEDICION = ("B5:conteo_inconsistente",)
FACTORES_B1 = {
    "categoria": "category",
    "formato": "store_format",
    "region": "region",
    "nse": "socioeconomic_level",
    "dia_semana": "day_of_week",
    "mes": "month",
}
BANDERAS = {
    "festivo": "is_holiday",
    "quincena": "is_payday",
    "temporada_navidena": "is_navidad_season",
    "buen_fin": "is_buen_fin",
    "semana_santa": "is_semana_santa",
}
# Las 13 dimensiones de B2
DIMENSIONES: dict[str, tuple[str, ...]] = {
    "serie": SERIE,
    "tienda": ("store_id",),
    **{nombre: (columna,) for nombre, columna in FACTORES_B1.items()},
    **{nombre: (columna,) for nombre, columna in BANDERAS.items()},
}
DESFASES_B4 = (1, 7)


def holm(p_valores: Sequence[float]) -> list[float]:
    """Los valores p ajustados por Holm (`statsmodels.stats.multitest.multipletests`)."""
    if not p_valores:
        return []
    return [float(p) for p in multipletests(list(p_valores), alpha=ALFA_HOLM, method="holm")[1]]


def _chi2(tabla: np.ndarray) -> tuple[float, int, float]:
    """χ² de Pearson de homogeneidad de una tabla de conteos: estadístico, gl y p."""
    r = stats.chi2_contingency(np.asarray(tabla, dtype=float), correction=False)
    return float(r.statistic), int(r.dof), float(r.pvalue)


def omnibus(m: pl.DataFrame) -> dict[str, Any]:
    """B1: modelo lineal de probabilidad de la promoción sobre categoría, formato, región, NSE, día de la
    semana, mes (indicadoras, sin el primer nivel) y las cinco banderas de calendario. Wald conjunta de que
    todas las pendientes son cero, agrupada por tienda."""
    columnas = [pl.lit(1.0).alias("constante")]
    for nombre, columna in FACTORES_B1.items():
        for nivel in sorted(m[columna].unique().to_list())[1:]:
            columnas.append((pl.col(columna) == nivel).cast(pl.Float64).alias(f"{nombre}={nivel}"))
    columnas += [pl.col(c).cast(pl.Float64).alias(nombre) for nombre, c in BANDERAS.items()]
    x = m.select(PROMO, "store_id", *columnas)
    nombres = x.columns[2:]
    aj = estimar(x, PROMO, nombres, efectos=())
    w = wald(aj, np.eye(len(nombres))[1:])
    return {
        "prueba": "B1",
        "descripcion": "MLP de has_promotion sobre segmento y calendario; Wald de pendientes nulas",
        "n": aj.n,
        "pendientes": len(nombres) - 1,
        "f": w.f,
        "gl": [w.gl_num, w.gl_den],
        "p_valor": w.p_valor,
        "p_sin_correccion": w.p_sin_correccion,
        "umbral": ALFA_CONJUNTA,
        "cumple": w.p_valor >= ALFA_CONJUNTA,
    }


def homogeneidad(m: pl.DataFrame, nombre: str, columnas: Sequence[str]) -> dict[str, Any]:
    """B2: χ² de homogeneidad de la tasa de promoción entre los niveles de una dimensión. Las tasas van
    resumidas en su mínimo y su máximo."""
    tabla = (
        m.group_by(columnas)
        .agg((pl.col(PROMO) == 0).sum().alias("sin"), (pl.col(PROMO) == 1).sum().alias("con"))
        .sort(columnas)
    )
    chi2, gl, p = _chi2(tabla.select("sin", "con").to_numpy())
    tasa = (tabla["con"] / (tabla["con"] + tabla["sin"]) * 100).to_numpy()
    return {
        "prueba": f"B2:{nombre}",
        "niveles": tabla.height,
        "chi2": chi2,
        "gl": gl,
        "p_valor": p,
        "tasa_min_pct": float(tasa.min()),
        "tasa_max_pct": float(tasa.max()),
    }


def ventas_previas(m: pl.DataFrame, adelantos: Sequence[str], y: str) -> dict[str, Any]:
    """B3: log(unidades) de t sobre la promoción de t + 1 a t + 7, con los efectos fijos del modelo
    principal; Wald de que los siete coeficientes son cero, agrupada por tienda."""
    df = m.filter(pl.all_horizontal(pl.col(a).is_not_null() for a in adelantos))
    aj = estimar(df, y, adelantos)
    w = wald(aj, np.eye(len(adelantos)))
    betas = aj.res.params
    return {
        "prueba": "B3",
        "descripcion": (
            "log(units_sold) de t sobre la promoción de t+1 a t+7, con efectos fijos de serie y de fecha"
        ),
        "n": aj.n,
        "f": w.f,
        "gl": [w.gl_num, w.gl_den],
        "p_valor": w.p_valor,
        "p_sin_correccion": w.p_sin_correccion,
        "beta_max_abs": float(np.max(np.abs(betas))),
        "umbral": ALFA_CONJUNTA,
        "cumple": w.p_valor >= ALFA_CONJUNTA,
    }


def _dos_por_dos(df: pl.DataFrame, a: pl.Expr, b: pl.Expr) -> np.ndarray:
    conteos = df.select(((a == i) & (b == j)).sum().alias(f"{i}{j}") for i in (0, 1) for j in (0, 1)).row(0)
    return np.array(conteos, dtype=float).reshape(2, 2)


def independencia_serial(preparado: pl.DataFrame, k: int, rezago: str) -> dict[str, Any]:
    """B4: χ² de la tabla 2 × 2 de (promoción en t, promoción en t − k), con los pares de cada serie."""
    df = preparado.filter(pl.col(PROMO).is_not_null() & pl.col(rezago).is_not_null())
    tabla = _dos_por_dos(df, pl.col(rezago), pl.col(PROMO))
    chi2, gl, p = _chi2(tabla)
    return {
        "prueba": f"B4:t-{k}",
        "n": int(tabla.sum()),
        "chi2": chi2,
        "gl": gl,
        "p_valor": p,
        "tasa_si_hubo_pct": float(100 * tabla[1, 1] / tabla[1].sum()),
        "tasa_si_no_hubo_pct": float(100 * tabla[0, 1] / tabla[0].sum()),
    }


def atricion(preparado: pl.DataFrame) -> list[dict[str, Any]]:
    """B5: χ² de 2 × 2 entre la promoción y que `units_sold` sea nulo (filas existentes), y entre la promoción
    y `flag_conteo_inconsistente` (filas con `cash_transactions`, donde la bandera se puede evaluar)."""
    existentes = preparado.filter(~pl.col("flag_fila_faltante"))
    casos = {
        "units_sold_nulo": (existentes, pl.col("units_sold").is_null()),
        "conteo_inconsistente": (
            existentes.filter(pl.col("cash_transactions").is_not_null()),
            pl.col("flag_conteo_inconsistente"),
        ),
    }
    salida = []
    for nombre, (df, evento) in casos.items():
        tabla = _dos_por_dos(df, pl.col(PROMO), evento.cast(pl.Int64))
        chi2, gl, p = _chi2(tabla)
        salida.append(
            {
                "prueba": f"B5:{nombre}",
                "n": int(tabla.sum()),
                "chi2": chi2,
                "gl": gl,
                "p_valor": p,
                "tasa_con_promocion_pct": float(100 * tabla[1, 1] / tabla[1].sum()),
                "tasa_sin_promocion_pct": float(100 * tabla[0, 1] / tabla[0].sum()),
            }
        )
    return salida


def evaluar(
    preparado: pl.DataFrame,
    m: pl.DataFrame,
    rezagos: Sequence[str],
    adelantos: Sequence[str],
    y: str,
) -> dict[str, Any]:
    """Las cinco pruebas, Holm sobre la familia y la regla. `rezagos` son las columnas de la promoción en
    t − 1 y t − 7 (en el orden de `DESFASES_B4`) y `adelantos`, las de t + 1 a t + 7."""
    familia = [homogeneidad(m, nombre, columnas) for nombre, columnas in DIMENSIONES.items()]
    familia += [independencia_serial(preparado, k, r) for k, r in zip(DESFASES_B4, rezagos, strict=True)]
    familia += atricion(preparado)
    for prueba, p_holm in zip(familia, holm([f["p_valor"] for f in familia]), strict=True):
        prueba["p_holm"] = p_holm
    b1, b3 = omnibus(m), ventas_previas(m, adelantos, y)
    rechazadas = [f["prueba"] for f in familia if f["p_holm"] < ALFA_HOLM]
    fallan = [b["prueba"] for b in (b1, b3) if not b["cumple"]] + rechazadas
    return {
        "b1": b1,
        "familia": {
            "pruebas": familia,
            "n_pruebas": len(familia),
            "alfa": ALFA_HOLM,
            "p_holm_min": min(f["p_holm"] for f in familia),
            "rechazadas": rechazadas,
            "cumple": not rechazadas,
        },
        "b3": b3,
        "se_sostiene": not fallan,
        "fallan": fallan,
        "asignacion_se_sostiene": not [f for f in fallan if f not in MEDICION],
        "pruebas_de_medicion": list(MEDICION),
    }
