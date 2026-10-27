"""Potencia, inferencia por aleatorización y calibración de los errores, con permutaciones deterministas.

- Inferencia por aleatorización (Fisher): B permutaciones de la promoción entre las filas de una misma fecha,
  que conservan cuántas promociones hubo cada día. El estadístico es |β| y el p es
  (1 + #{|β_b| ≥ |β_obs|}) / (B + 1).
- Calibración A/A: otras permutaciones, cada una con su t agrupada por tienda. Si la tasa de
  |t| > t₀.₉₇₅,G−1 cae fuera del intervalo binomial del 99 % alrededor de 5 %, la inferencia principal pasa a
  ser la de aleatorización.
- Efecto mínimo detectable con α = 0.05 y potencia 0.8: (t₀.₉₇₅,gl + t₀.₈,gl) · EE, en % como e^MDE − 1.
- Diseño semanal: la promoción por tienda × categoría × semana con probabilidad 0.5, sobre las semanas
  completas del panel (lunes a domingo con los siete días observados) y el log de la suma semanal, con efectos
  fijos de serie y de semana. Su EE es la media de los EE agrupados de asignaciones placebo, en el panel
  completo y en ventanas reales de W semanas consecutivas.
- Semanas necesarias: el menor W de la rejilla cuyo MDE medido no pasa del efecto, o ninguna. No se
  extrapola con 1/√W: con pocas semanas el EE medido es mayor que la extrapolación, porque el efecto fijo de
  serie se come buena parte de la variación (con una sola semana, β no está identificado).

Las permutaciones se absorben en lotes de columnas: el lote cambia cuánta memoria se usa, no el orden de los
sorteos, y es fijo, así dos corridas dan lo mismo.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from demanda.causal.absorcion import PRINCIPAL, TIENDA, absorber, codigos, un_regresor
from demanda.datos.contratos import SERIE

LOTE = 25


@dataclass(frozen=True)
class Parametros:
    """Los sorteos, las semillas y los niveles de las pruebas, fijados en la spec."""

    permutaciones: int = 2_000
    semilla_fisher: int = 0
    placebos_aa: int = 500
    semilla_aa: int = 1
    placebos_semanal: int = 500
    semilla_semanal: int = 2
    probabilidad_semanal: float = 0.5
    semanas_rejilla: tuple[int, ...] = (2, 3, 4, 6, 8, 12)
    placebos_por_ventana: int = 100
    alfa: float = 0.05
    potencia: float = 0.8
    nivel_aa: float = 0.99
    efectos_semanal: tuple[float, ...] = (0.02, 0.05, 0.10)

    def a_json(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


class Diseno(NamedTuple):
    """Lo que necesitan las permutaciones: el resultado ya absorbido, la asignación, los factores de efectos
    fijos, la fecha (para permutar dentro de ella) y la tienda (para agrupar)."""

    y: np.ndarray
    d: np.ndarray
    factores: list[np.ndarray]
    fecha: np.ndarray
    tienda: np.ndarray


def diseno(m: pl.DataFrame, y: str, promo: str) -> Diseno:
    factores = [codigos(m, e) for e in PRINCIPAL]
    return Diseno(
        absorber(m[y].to_numpy(), factores).datos[:, 0],
        m[promo].to_numpy().astype(float),
        factores,
        codigos(m, ("date",)),
        codigos(m, TIENDA),
    )


def permutador(d: np.ndarray, fecha: np.ndarray) -> Callable[[np.random.Generator], np.ndarray]:
    """Una función que devuelve una permutación de `d` entre las filas de cada fecha: cada día conserva sus
    promociones.

    Los valores van en una matriz fecha × posición rellena con NaN, y `Generator.permuted` baraja cada
    renglón. Los valores reales de un renglón barajado, leídos en orden, son una permutación uniforme de los
    de esa fecha, y vuelven a las filas de la fecha en su orden original. Así no se ordena nada por sorteo."""
    if np.isnan(d).any():
        raise ValueError("la asignación a permutar no puede tener nulos")
    orden = np.argsort(fecha, kind="stable")
    conteos = np.bincount(fecha)
    inicio = np.repeat(np.cumsum(conteos) - conteos, conteos)
    base = np.full((conteos.size, int(conteos.max())), np.nan)
    base[fecha[orden], np.arange(d.size) - inicio] = d[orden]

    def permutar(rng: np.random.Generator) -> np.ndarray:
        barajado = rng.permuted(base, axis=1)
        salida = np.empty(d.size)
        salida[orden] = barajado[~np.isnan(barajado)]
        return salida

    return permutar


def placebos(
    y: np.ndarray,
    factores: Sequence[np.ndarray],
    grupos: np.ndarray,
    asignar: Callable[[], np.ndarray],
    n: int,
    lote: int = LOTE,
) -> tuple[np.ndarray, np.ndarray]:
    """β y t agrupada de `y` (ya absorbido) sobre `n` asignaciones de `asignar`, absorbidas en lotes."""
    betas, ts = [], []
    for inicio in range(0, n, lote):
        x = np.column_stack([asignar() for _ in range(min(lote, n - inicio))])
        b, t = un_regresor(y, absorber(x, factores).datos, grupos)
        betas.append(b)
        ts.append(t)
    return np.concatenate(betas), np.concatenate(ts)


def p_fisher(beta_obs: float, betas: np.ndarray) -> float:
    """(1 + #{|β_b| ≥ |β_obs|}) / (B + 1): el mínimo posible es 1 / (B + 1)."""
    return float((1 + np.sum(np.abs(betas) >= abs(beta_obs))) / (betas.size + 1))


def aleatorizacion(dis: Diseno, beta_obs: float, p: Parametros) -> dict[str, Any]:
    """La inferencia por aleatorización del efecto principal."""
    rng = np.random.default_rng(p.semilla_fisher)
    permutar = permutador(dis.d, dis.fecha)
    betas, _ = placebos(dis.y, dis.factores, dis.tienda, lambda: permutar(rng), p.permutaciones)
    return {
        "descripcion": "permutaciones de has_promotion entre las filas de la misma fecha",
        "permutaciones": p.permutaciones,
        "semilla": p.semilla_fisher,
        "estadistico": "|β|",
        "beta_obs": beta_obs,
        "superan": int(np.sum(np.abs(betas) >= abs(beta_obs))),
        "p_valor": p_fisher(beta_obs, betas),
        "p_minimo": 1 / (p.permutaciones + 1),
        "de_beta_placebo": float(np.std(betas, ddof=1)),
    }


def intervalo_aa(n: int, alfa: float, nivel: float) -> tuple[float, float]:
    """El intervalo central de `nivel` de la tasa de rechazos de n pruebas a nivel α, si están calibradas."""
    inf, sup = stats.binom.interval(nivel, n, alfa)
    return float(inf) / n, float(sup) / n


def calibracion_aa(dis: Diseno, p: Parametros) -> dict[str, Any]:
    """La tasa de rechazo de la t agrupada en asignaciones placebo, que no tienen efecto."""
    rng = np.random.default_rng(p.semilla_aa)
    permutar = permutador(dis.d, dis.fecha)
    betas, ts = placebos(dis.y, dis.factores, dis.tienda, lambda: permutar(rng), p.placebos_aa)
    gl = int(dis.tienda.max())  # G − 1
    critico = float(stats.t.ppf(1 - p.alfa / 2, gl))
    rechazos = int(np.sum(np.abs(ts) > critico))
    inf, sup = intervalo_aa(p.placebos_aa, p.alfa, p.nivel_aa)
    tasa = rechazos / p.placebos_aa
    dentro = inf <= tasa <= sup
    return {
        "descripcion": "permutaciones de has_promotion en la fecha, cada una con su t agrupada por tienda",
        "placebos": p.placebos_aa,
        "semilla": p.semilla_aa,
        "t_critica": critico,
        "gl": gl,
        "rechazos": rechazos,
        "tasa_pct": 100 * tasa,
        "intervalo_99_pct": [100 * inf, 100 * sup],
        "dentro": dentro,
        "ee_agrupado_medio": float(np.mean(np.abs(betas / ts))),
        "de_beta_placebo": float(np.std(betas, ddof=1)),
        "inferencia_principal": "agrupada_por_tienda" if dentro else "aleatorizacion",
    }


def factor_mde(gl: int, alfa: float = 0.05, potencia: float = 0.8) -> float:
    return float(stats.t.ppf(1 - alfa / 2, gl) + stats.t.ppf(potencia, gl))


def mde(ee: float, gl: int, alfa: float = 0.05, potencia: float = 0.8) -> dict[str, Any]:
    """El efecto mínimo detectable de un coeficiente con error estándar `ee`, en log y en %."""
    valor = factor_mde(gl, alfa, potencia) * ee
    return {"ee": ee, "mde_log": valor, "mde_pct": 100 * math.expm1(valor)}


def panel_semanal(m: pl.DataFrame) -> pl.DataFrame:
    """Las semanas completas de cada serie (lunes a domingo, con los siete días en la muestra) y el log de su
    suma de unidades."""
    return (
        m.with_columns(pl.col("date").dt.truncate("1w").alias("semana"))
        .group_by(*SERIE, "semana")
        .agg(pl.len().alias("dias"), pl.col("units_sold").sum().alias("unidades"))
        .filter(pl.col("dias") == 7)
        .with_columns(pl.col("unidades").cast(pl.Float64).log().alias("log_unidades"))
        .sort(*SERIE, "semana")
    )


def placebos_semanales(
    s: pl.DataFrame, rng: np.random.Generator, n: int, probabilidad: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """β, t agrupada por tienda y gl (G − 1) de `n` asignaciones al azar por serie-semana de `s`, con efectos
    fijos de serie y de semana."""
    factores = [codigos(s, SERIE), codigos(s, ("semana",))]
    y = absorber(s["log_unidades"].to_numpy(), factores).datos[:, 0]
    tienda = codigos(s, TIENDA)
    betas, ts = placebos(y, factores, tienda, lambda: (rng.random(s.height) < probabilidad).astype(float), n)
    return betas, ts, int(tienda.max())


def ventanas(lunes: Sequence[date], w: int) -> list[list[date]]:
    """Las ventanas no traslapadas de `w` semanas consecutivas, desde la primera; la cola que no completa
    una ventana queda fuera."""
    return [list(lunes[i : i + w]) for i in range(0, len(lunes) - w + 1, w)]


def por_semanas(s: pl.DataFrame, rng: np.random.Generator, p: Parametros) -> list[dict[str, Any]]:
    """Por cada W de la rejilla: el EE medio de los placebos en todas las ventanas de W semanas del panel
    semanal, con el menor gl de las ventanas, y su MDE. Los W sin ninguna ventana completa no aparecen."""
    lunes = s["semana"].unique().sort().to_list()
    filas = []
    for w in p.semanas_rejilla:
        ees, gls = [], []
        for semanas in ventanas(lunes, w):
            sub = s.filter(pl.col("semana").is_in(semanas))
            betas, ts, gl = placebos_semanales(sub, rng, p.placebos_por_ventana, p.probabilidad_semanal)
            ees.append(float(np.mean(np.abs(betas / ts))))
            gls.append(gl)
        if ees:
            ee, gl = float(np.mean(ees)), min(gls)
            filas.append({"semanas": w, "ventanas": len(ees), "gl": gl, **mde(ee, gl, p.alfa, p.potencia)})
    return filas


def semanas_necesarias(tabla: Sequence[Mapping[str, Any]], efecto: float) -> int | None:
    """El menor W de la tabla cuyo MDE no pasa de `efecto` (fracción): mde_log ≤ log(1 + efecto). None si
    ninguno lo alcanza."""
    return next(
        (
            f["semanas"]
            for f in sorted(tabla, key=lambda f: f["semanas"])
            if f["mde_log"] <= math.log1p(efecto)
        ),
        None,
    )


def diseno_semanal(m: pl.DataFrame, p: Parametros) -> dict[str, Any]:
    """El EE del diseño semanal propuesto, medido con asignaciones placebo: en el panel semanal completo y en
    ventanas de W semanas. Un solo generador, primero para el panel completo y después para las ventanas."""
    s = panel_semanal(m)
    rng = np.random.default_rng(p.semilla_semanal)
    betas, ts, gl = placebos_semanales(s, rng, p.placebos_semanal, p.probabilidad_semanal)
    ee = float(np.mean(np.abs(betas / ts)))
    critico = float(stats.t.ppf(1 - p.alfa / 2, gl))
    tabla = por_semanas(s, rng, p)
    return {
        "descripcion": (
            "promoción por tienda × categoría × semana con probabilidad 0.5; log de la suma semanal, efectos "
            "fijos de serie y de semana, errores agrupados por tienda"
        ),
        "placebos": p.placebos_semanal,
        "semilla": p.semilla_semanal,
        "semanas": s["semana"].n_unique(),
        "series_semana": s.height,
        "series": s.select(SERIE).n_unique(),
        "desde": str(s["semana"].min()),  # el lunes de la primera semana y el domingo de la última
        "hasta": str(s.select(pl.col("semana").max() + timedelta(days=6)).item()),
        "ee": ee,
        "de_beta_placebo": float(np.std(betas, ddof=1)),
        "tasa_rechazo_placebo_pct": 100 * float(np.mean(np.abs(ts) > critico)),
        "mde_actual": mde(ee, gl, p.alfa, p.potencia),
        "placebos_por_ventana": p.placebos_por_ventana,
        "por_semanas": tabla,
        "semanas_necesarias": [
            {"efecto_pct": 100 * e, "semanas": semanas_necesarias(tabla, e)} for e in p.efectos_semanal
        ],
        "metodo": (
            "EE medido con asignaciones placebo en todas las ventanas no traslapadas de W semanas "
            "consecutivas del panel semanal, promediado entre ventanas; semanas necesarias: el menor W de la "
            "rejilla con MDE ≤ efecto, o nulo si ninguno lo alcanza"
        ),
    }
