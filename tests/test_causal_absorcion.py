"""La absorción de efectos fijos y los errores agrupados contra indicadoras, a mano y contra statsmodels."""

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from scipy import stats

from demanda.causal.absorcion import (
    absorber,
    ajustar,
    codigos,
    coeficiente,
    combinacion,
    estimar,
    igualdad,
    un_regresor,
    wald,
)


def _panel_chico(semilla: int = 0) -> pl.DataFrame:
    """6 series (2 por tienda) × 15 fechas con ~20 % de filas fuera: un panel desbalanceado con efectos de
    serie y de fecha, dos regresores y pesos."""
    rng = np.random.default_rng(semilla)
    filas = [(s, t) for s in range(6) for t in range(15) if rng.random() > 0.2]
    s, t = (np.array(c) for c in zip(*filas, strict=True))
    x1, x2 = rng.normal(size=s.size) + 0.3 * s, rng.binomial(1, 0.3, s.size).astype(float)
    y = 0.5 * x1 - 0.2 * x2 + rng.normal(0, 1, 6)[s] + rng.normal(0, 1, 15)[t] + rng.normal(0, 0.5, s.size)
    return pl.DataFrame(
        {
            "serie": s,
            "fecha": t,
            "tienda": s // 2,
            "x1": x1,
            "x2": x2,
            "y": y,
            "peso": rng.uniform(1, 5, s.size),
        }
    )


def _indicadoras(df: pl.DataFrame) -> np.ndarray:
    """Todas las de serie y las de fecha sin la primera: sin constante, el rango es completo."""
    series = (df["serie"].to_numpy()[:, None] == np.arange(6)).astype(float)
    fechas = (df["fecha"].to_numpy()[:, None] == np.arange(1, 15)).astype(float)
    return np.column_stack([series, fechas])


def test_beta_absorbido_es_el_de_las_indicadoras():
    df = _panel_chico()
    x = df.select("x1", "x2").to_numpy()
    completo = sm.OLS(df["y"].to_numpy(), np.column_stack([x, _indicadoras(df)])).fit()
    aj = estimar(df, "y", ["x1", "x2"], efectos=(("serie",), ("fecha",)), cluster=(("tienda",),))
    assert aj.iteraciones > 1  # desbalanceado: no converge en una pasada
    np.testing.assert_allclose(aj.res.params, completo.params[:2], rtol=0, atol=1e-10)
    # Ponderado: la absorción ponderada más WLS da lo mismo que WLS con indicadoras
    w = df["peso"].to_numpy()
    completo_w = sm.WLS(df["y"].to_numpy(), np.column_stack([x, _indicadoras(df)]), weights=w).fit()
    aj_w = estimar(
        df, "y", ["x1", "x2"], efectos=(("serie",), ("fecha",)), cluster=(("tienda",),), pesos="peso"
    )
    np.testing.assert_allclose(aj_w.res.params, completo_w.params[:2], rtol=0, atol=1e-10)


def test_el_error_agrupado_es_el_sandwich_a_mano():
    df = _panel_chico()
    aj = estimar(df, "y", ["x1", "x2"], efectos=(("serie",), ("fecha",)), cluster=(("tienda",),))
    datos = absorber(
        df.select("y", "x1", "x2").to_numpy(), [df["serie"].to_numpy(), df["fecha"].to_numpy()]
    ).datos
    y, x = datos[:, 0], datos[:, 1:]
    pan = np.linalg.inv(x.T @ x)
    u = y - x @ (pan @ x.T @ y)
    tienda = df["tienda"].to_numpy()
    carne = sum(
        np.outer(x[tienda == g].T @ u[tienda == g], x[tienda == g].T @ u[tienda == g]) for g in range(3)
    )
    n, k, g = df.height, 2, 3
    v = g / (g - 1) * (n - 1) / (n - k) * pan @ carne @ pan
    np.testing.assert_allclose(aj.res.bse, np.sqrt(np.diag(v)), rtol=1e-10)
    # La t lleva G − 1 grados de libertad y el IC95 sale de ella
    e = coeficiente(aj, "x1")
    assert e.gl == aj.gl == 2
    assert e.ic_sup - e.valor == pytest.approx(stats.t.ppf(0.975, 2) * np.sqrt(v[0, 0]), rel=1e-10)
    # Una combinación lineal usa la covarianza completa
    suma = combinacion(aj, {"x1": 1.0, "x2": 1.0})
    assert suma.valor == pytest.approx(aj.res.params.sum(), abs=1e-14)
    assert suma.ee == pytest.approx(np.sqrt(v.sum()), rel=1e-10)


def test_un_regresor_es_ajustar_con_un_regresor():
    df = _panel_chico(semilla=3)
    factores = [codigos(df, ("serie",)), codigos(df, ("fecha",))]
    datos = absorber(df.select("y", "x1", "x2").to_numpy(), factores).datos
    tienda = df["tienda"].to_numpy()
    betas, ts = un_regresor(datos[:, 0], datos[:, 1:], tienda)
    for j, nombre in enumerate(("x1", "x2")):
        aj = ajustar(datos[:, 0], datos[:, 1 + j], [nombre], tienda)
        assert betas[j] == pytest.approx(aj.res.params[0], rel=1e-12)
        assert ts[j] == pytest.approx(aj.res.tvalues[0], rel=1e-10)


def test_igualdad_y_colinealidad():
    df = _panel_chico(semilla=5).with_columns((pl.col("x1") * 2).alias("x1_doble"))
    aj = estimar(df, "y", ["x1", "x2"], efectos=(("serie",), ("fecha",)), cluster=(("tienda",),))
    w = igualdad(aj, ["x1", "x2"])
    # Con dos coeficientes, la F de igualdad es el cuadrado de la t de la diferencia
    diferencia = combinacion(aj, {"x1": 1.0, "x2": -1.0})
    assert w.f == pytest.approx(diferencia.t**2, rel=1e-10)
    assert (w.gl_num, w.gl_den) == (1, 2)
    # Con q = 1 la corrección de Hotelling no cambia nada: los dos p son el de la t
    assert w.p_valor == pytest.approx(diferencia.p_valor, rel=1e-8)
    assert w.p_sin_correccion == pytest.approx(diferencia.p_valor, rel=1e-8)
    # Con tres tiendas no se pueden probar tres restricciones a la vez
    with pytest.raises(ValueError, match="no tiene rango"):
        wald(aj, np.eye(2)[[0, 1, 0]])
    with pytest.raises(ValueError, match="colineales"):
        estimar(df, "y", ["x1", "x1_doble"], efectos=(("serie",),), cluster=(("tienda",),))
    # Un regresor constante dentro de la serie desaparece al absorberla
    with pytest.raises(ValueError, match="absorbe"):
        estimar(df, "y", ["x1", "tienda"], efectos=(("serie",),), cluster=(("tienda",),))


def test_absorber_deja_los_residuos_de_las_indicadoras():
    df = _panel_chico(semilla=7)
    indicadoras = _indicadoras(df)
    x = df.select("y", "x1", "x2").to_numpy()
    for pesos in (None, df["peso"].to_numpy()):
        datos = absorber(x, [df["serie"].to_numpy(), df["fecha"].to_numpy()], pesos).datos
        raiz = np.ones(df.height) if pesos is None else np.sqrt(pesos)
        coef = np.linalg.lstsq(indicadoras * raiz[:, None], x * raiz[:, None], rcond=None)[0]
        np.testing.assert_allclose(datos, x - indicadoras @ coef, rtol=0, atol=1e-9)
