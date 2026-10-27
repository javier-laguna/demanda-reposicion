"""Estadísticos del monitor como funciones puras: PSI y pruebas binomiales.

- PSI: Σ (a − e) · ln(a / e) sobre intervalos definidos por los cuantiles de la referencia, con e las
  proporciones de la referencia y a las de la semana, cada una con un piso. Los cortes son valores de la
  referencia (cuantiles sin interpolar), así que el PSI no cambia ante una transformación creciente, como el
  logaritmo: la razón entra sin log.
- Binomiales: `scipy.stats.binomtest`, de una cola (más casos de los esperados) o de dos.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

INTERVALOS = 10
PISO = 1e-4


def cortes(referencia: Sequence[float] | np.ndarray, intervalos: int = INTERVALOS) -> np.ndarray:
    """Los `intervalos − 1` cortes interiores: los cuantiles k / intervalos de la referencia, sin
    interpolar."""
    probabilidades = np.arange(1, intervalos) / intervalos
    return np.quantile(np.asarray(referencia, dtype=float), probabilidades, method="inverted_cdf")


def proporciones(
    valores: Sequence[float] | np.ndarray, limites: np.ndarray, piso: float = PISO
) -> np.ndarray:
    """La proporción de valores en cada intervalo (−∞, c₁], (c₁, c₂], …, (c_k, ∞), con piso."""
    x = np.asarray(valores, dtype=float)
    conteos = np.bincount(np.searchsorted(limites, x, side="left"), minlength=limites.size + 1)
    return np.maximum(conteos / x.size, piso)


def psi(
    referencia: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    intervalos: int = INTERVALOS,
    piso: float = PISO,
) -> float | None:
    """El PSI de `actual` contra `referencia`, con intervalos en los cuantiles de la referencia. Sin datos en
    alguna de las dos, None."""
    ref, act = np.asarray(referencia, dtype=float), np.asarray(actual, dtype=float)
    if not ref.size or not act.size:
        return None
    limites = cortes(ref, intervalos)
    e, a = proporciones(ref, limites, piso), proporciones(act, limites, piso)
    return float(np.sum((a - e) * np.log(a / e)))


def binomial(casos: int, n: int, p: float, colas: int) -> float:
    """El valor p de observar `casos` de `n` con probabilidad `p`: de una cola (mayor) o de dos.

    Sin observaciones no hay evidencia (1.0). Con p en 0 o 1, todo resultado distinto del único posible es
    imposible (0.0).
    """
    if n == 0:
        return 1.0
    if p <= 0 or p >= 1:
        posible = 0 if p <= 0 else n
        return 1.0 if casos == posible or (colas == 1 and casos < posible) else 0.0
    alternativa = "greater" if colas == 1 else "two-sided"
    return float(stats.binomtest(casos, n, p, alternative=alternativa).pvalue)
