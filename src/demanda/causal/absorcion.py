"""Efectos fijos por proyecciones alternadas y regresión con errores agrupados, como funciones puras.

- `absorber` resta por turnos la media de cada grupo de cada factor (ponderada, si hay pesos), hasta que el
  cambio máximo de una pasada completa baje de `TOL`. Converge a la proyección sobre el complemento de las
  indicadoras de todos los factores juntos: por Frisch-Waugh-Lovell, β sobre las variables transformadas es
  el mismo que con las indicadoras.
- `ajustar` corre `statsmodels` OLS (WLS con pesos) sobre las variables transformadas, sin constante, con
  errores agrupados CRV1. El factor de muestra finita es G / (G − 1) · (N − 1) / (N − K), con K solo los
  regresores que no se absorbieron, y la t lleva G − 1 grados de libertad: la convención de `reghdfe` cuando
  los efectos fijos están anidados en los clústeres. Con dos columnas de grupos la agrupación es doble
  (Cameron, Gelbach y Miller) y G es el menor de los dos.
- `wald` prueba q restricciones con la covarianza agrupada. La F = W / q contra F(q, G − 1) rechaza de más
  cuando q no es chico frente a G, porque la covarianza agrupada tiene a lo más G − 1 grados de libertad.
  Por eso el p usa la corrección de Hotelling, F · (G − q) / (G − 1) contra F(q, G − q), y el de
  F(q, G − 1) queda como `p_sin_correccion`. Con q = 1 coinciden con la t.
- `un_regresor` hace el mismo cálculo con numpy para un regresor y muchas variables a la vez: lo usan las
  permutaciones, que necesitan miles de ajustes.
- `estimar` arma las tres cosas desde una tabla: absorbe, ajusta y agrupa por las columnas que se le den.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import sparse, stats

from demanda.datos.contratos import SERIE

TOL = 1e-10
MAX_ITER = 1_000
ABSORBIDO = 1e-7  # norma tras absorber / norma original por debajo de la cual el regresor desaparece
# Los efectos fijos del modelo principal (serie y fecha) y el clúster (tienda), como columnas del panel
PRINCIPAL: tuple[tuple[str, ...], ...] = (SERIE, ("date",))
TIENDA: tuple[str, ...] = ("store_id",)


class Absorcion(NamedTuple):
    datos: np.ndarray  # (n, k): las variables menos sus efectos fijos
    iteraciones: int  # pasadas completas hasta converger


class Ajuste(NamedTuple):
    res: Any  # el resultado de statsmodels, con la covarianza agrupada
    nombres: tuple[str, ...]  # de los regresores, en el orden de las columnas
    iteraciones: int  # de la absorción

    @property
    def n(self) -> int:
        return int(self.res.nobs)

    @property
    def gl(self) -> int:
        """Los grados de libertad de la t y de la F: G − 1."""
        return int(self.res.df_resid_inference)


class Estimacion(NamedTuple):
    """Un coeficiente o una combinación lineal de coeficientes, con su IC95 de la t con G − 1 gl."""

    valor: float
    ee: float
    t: float
    gl: int
    p_valor: float
    ic_inf: float
    ic_sup: float


class Wald(NamedTuple):
    f: float  # W / q
    gl_num: int  # q
    gl_den: int  # G − q, de la corrección de Hotelling
    p_valor: float  # F · (G − q) / (G − 1) contra F(q, G − q)
    p_sin_correccion: float  # F contra F(q, G − 1)


def densos(codigos: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Los códigos como enteros consecutivos desde 0, en el orden de sus valores."""
    return np.unique(np.asarray(codigos), return_inverse=True)[1].ravel()


def sumador(codigos: np.ndarray, pesos: np.ndarray | None = None) -> sparse.csr_matrix:
    """La matriz G × n que suma por grupo cada columna, ponderada por `pesos`. Los códigos son densos."""
    n = codigos.size
    w = np.ones(n) if pesos is None else np.asarray(pesos, dtype=float)
    return sparse.csr_matrix((w, (codigos, np.arange(n))), shape=(int(codigos.max()) + 1, n))


def absorber(
    x: np.ndarray,
    factores: Sequence[np.ndarray],
    pesos: np.ndarray | None = None,
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> Absorcion:
    """`x` (n o n × k) menos sus efectos fijos de `factores`, por proyecciones alternadas."""
    datos = np.array(x, dtype=float)
    if datos.ndim == 1:
        datos = datos[:, None]
    if not factores:
        return Absorcion(datos, 0)
    w = np.ones(datos.shape[0]) if pesos is None else np.asarray(pesos, dtype=float)
    proyectores = []
    for factor in factores:
        f = densos(factor)
        proyectores.append((f, sumador(f, w), np.bincount(f, weights=w)))
    for iteracion in range(1, max_iter + 1):
        previo = datos.copy()
        for f, suma, masa in proyectores:
            datos -= ((suma @ datos) / masa[:, None])[f]
        if np.max(np.abs(datos - previo)) < tol:
            return Absorcion(datos, iteracion)
    raise ValueError(f"la absorción de efectos fijos no convergió en {max_iter} pasadas")


def ajustar(
    y: np.ndarray,
    x: np.ndarray,
    nombres: Sequence[str],
    grupos: np.ndarray,
    pesos: np.ndarray | None = None,
    iteraciones: int = 0,
) -> Ajuste:
    """OLS (o WLS) de `y` sobre `x`, ya transformadas, sin constante y con errores agrupados por `grupos`
    (un vector, o dos columnas para la agrupación doble)."""
    exog = np.asarray(x, dtype=float).reshape(len(y), -1)
    if exog.shape[1] != len(nombres):
        raise ValueError("hay un nombre por columna de x")
    if np.linalg.matrix_rank(exog) < exog.shape[1]:
        raise ValueError(f"los regresores {', '.join(nombres)} son colineales tras absorber")
    modelo = sm.OLS(y, exog) if pesos is None else sm.WLS(y, exog, weights=pesos)
    res = modelo.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(grupos)}, use_t=True)
    return Ajuste(res, tuple(nombres), iteraciones)


def combinacion(aj: Ajuste, pesos: Mapping[str, float]) -> Estimacion:
    """Σ w_j · β_j con su error agrupado; un solo nombre con peso 1 es el coeficiente."""
    r = np.zeros(len(aj.nombres))
    for nombre, w in pesos.items():
        r[aj.nombres.index(nombre)] = w
    prueba = aj.res.t_test(r)
    inf, sup = np.asarray(prueba.conf_int(alpha=0.05)).ravel()
    return Estimacion(
        float(np.ravel(prueba.effect)[0]),
        float(np.ravel(prueba.sd)[0]),
        float(np.ravel(prueba.tvalue)[0]),
        int(prueba.df_denom),
        float(np.ravel(prueba.pvalue)[0]),
        float(inf),
        float(sup),
    )


def coeficiente(aj: Ajuste, nombre: str) -> Estimacion:
    return combinacion(aj, {nombre: 1.0})


def wald(aj: Ajuste, restricciones: np.ndarray) -> Wald:
    """F de Wald de R · β = 0 con la covarianza agrupada, con y sin la corrección de Hotelling."""
    r = np.atleast_2d(np.asarray(restricciones, dtype=float))
    g = aj.gl + 1
    if r.shape[0] >= g:
        # La covarianza agrupada tiene rango G − 1 a lo más: con q ≥ G, statsmodels bajaría q al rango
        raise ValueError(f"{r.shape[0]} restricciones con {g} clústeres: la Wald agrupada no tiene rango")
    prueba = aj.res.wald_test(r, use_f=True, scalar=True)
    f, q, g_menos_1 = float(prueba.fvalue), int(prueba.df_num), int(prueba.df_denom)
    if q != r.shape[0]:
        raise ValueError(f"la covarianza agrupada de las {r.shape[0]} restricciones tiene rango {q}")
    corregida = float(stats.f.sf(f * (g_menos_1 + 1 - q) / g_menos_1, q, g_menos_1 + 1 - q))
    return Wald(f, q, g_menos_1 + 1 - q, corregida, float(prueba.pvalue))


def igualdad(aj: Ajuste, nombres: Sequence[str]) -> Wald:
    """F de Wald de que los coeficientes de `nombres` son iguales: β_1 − β_j = 0 para j = 2..k."""
    r = np.zeros((len(nombres) - 1, len(aj.nombres)))
    primero = aj.nombres.index(nombres[0])
    for i, nombre in enumerate(nombres[1:]):
        r[i, primero], r[i, aj.nombres.index(nombre)] = 1.0, -1.0
    return wald(aj, r)


def un_regresor(y: np.ndarray, x: np.ndarray, grupos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """β y t agrupada de `y` sobre cada columna de `x` por separado, sin constante y ya transformadas. Es lo
    mismo que `ajustar` con un regresor: con K = 1, el factor de muestra finita es G / (G − 1)."""
    xs = np.asarray(x, dtype=float)
    if xs.ndim == 1:
        xs = xs[:, None]
    sxx = np.einsum("ij,ij->j", xs, xs)
    beta = (xs.T @ y) / sxx
    residuos = y[:, None] - xs * beta
    suma = sumador(densos(grupos))
    puntajes = suma @ (xs * residuos)
    g = suma.shape[0]
    varianza = g / (g - 1) * np.einsum("ij,ij->j", puntajes, puntajes) / sxx**2
    return beta, beta / np.sqrt(varianza)


def codigos(df: pl.DataFrame, columnas: Sequence[str]) -> np.ndarray:
    """Un entero por combinación de valores de `columnas`, de 0 en adelante. Sin columnas, un solo grupo: su
    efecto fijo es la constante."""
    if not columnas:
        return np.zeros(df.height, dtype=np.int64)
    return df.select(pl.struct(*columnas).rank("dense")).to_series().to_numpy().astype(np.int64) - 1


def estimar(
    df: pl.DataFrame,
    y: str,
    regresores: Sequence[str],
    efectos: Sequence[Sequence[str]] = PRINCIPAL,
    cluster: Sequence[Sequence[str]] = (TIENDA,),
    pesos: str | None = None,
) -> Ajuste:
    """`y` sobre `regresores` con los efectos fijos de `efectos` absorbidos y errores agrupados por `cluster`
    (uno o dos), todo como columnas de `df`. Con `pesos`, la absorción y la regresión son ponderadas."""
    w = None if pesos is None else df[pesos].to_numpy().astype(float)
    x = df.select(y, *regresores).to_numpy().astype(float)
    ab = absorber(x, [codigos(df, e) for e in efectos], w)
    # Un regresor que los efectos fijos explican queda en residuos del tamaño de la tolerancia
    escala = np.linalg.norm(x[:, 1:], axis=0)
    absorbidos = [
        r
        for r, a, e in zip(regresores, np.linalg.norm(ab.datos[:, 1:], axis=0), escala, strict=True)
        if a <= ABSORBIDO * max(e, 1.0)
    ]
    if absorbidos:
        raise ValueError(f"los efectos fijos absorben {', '.join(absorbidos)}")
    grupos = np.column_stack([codigos(df, c) for c in cluster])
    return ajustar(
        ab.datos[:, 0],
        ab.datos[:, 1:],
        regresores,
        grupos[:, 0] if grupos.shape[1] == 1 else grupos,
        w,
        ab.iteraciones,
    )
