"""Diebold-Mariano y la pérdida por corte contra ejemplos calculados a mano."""

from datetime import date

import polars as pl
import pytest

from demanda.seleccion.diebold_mariano import diebold_mariano, errores_semanales, perdida_por_corte


def test_diebold_mariano_con_la_correccion_hln_a_mano():
    # d = 1, −1, 2, 0, 3: media 1 y desviaciones 0, −2, 1, −1, 2, con suma de cuadrados 10.
    # γ₀ = 10/5 = 2, var(d̄) = γ₀/n = 0.4 y DM = 1/√0.4 = 1.5811. HLN con h = 1: √((5 − 1)/5) = 0.8944,
    # así que DM* = √2 = 1.4142. Contra una t con 4 grados de libertad, p (dos colas) = 0.2302.
    dm = diebold_mariano([1, -1, 2, 0, 3])
    assert dm.n == 5
    assert dm.media == pytest.approx(1.0)
    assert dm.estadistico == pytest.approx(2**0.5)
    assert dm.p_valor == pytest.approx(0.2302, abs=1e-4)


def test_es_simetrico_al_intercambiar_a_y_b():
    d = [3.0, -1.0, 4.0, 1.5, -0.5, 2.0]
    ab, ba = diebold_mariano(d), diebold_mariano([-x for x in d])
    assert ba.estadistico == pytest.approx(-ab.estadistico)
    assert ba.media == pytest.approx(-ab.media)
    assert ba.p_valor == pytest.approx(ab.p_valor)


def test_rechaza_menos_de_tres_cortes_y_perdidas_constantes():
    with pytest.raises(ValueError, match="al menos 3"):
        diebold_mariano([1.0, 2.0])
    with pytest.raises(ValueError, match="no varían"):
        diebold_mariano([2.0, 2.0, 2.0])


O1, O2 = date(2023, 1, 5), date(2023, 1, 12)


def _pronosticos() -> pl.DataFrame:
    """2 modelos × 2 cortes × 2 tiendas, con 2 días por par. T2 en O2 tiene un día sin real."""
    filas = []
    # (origen, tienda, reales, pronóstico de A, pronóstico de B)
    pares = [
        (O1, "T1", (10, 20), (12, 20), (10, 25)),
        (O1, "T2", (5, 5), (5, 3), (6, 6)),
        (O2, "T1", (8, 8), (10, 10), (8, 4)),
        (O2, "T2", (None, 4), (1, 1), (2, 2)),
    ]
    for origen, tienda, y, a, b in pares:
        for modelo, yhat in (("A", a), ("B", b)):
            for h in (1, 2):
                filas.append((modelo, origen, tienda, "Ropa", h, y[h - 1], float(yhat[h - 1])))
    return pl.DataFrame(
        filas,
        schema={
            "modelo": pl.String,
            "origen": pl.Date,
            "store_id": pl.String,
            "category": pl.String,
            "h": pl.Int64,
            "y": pl.Float64,
            "yhat": pl.Float64,
        },
        orient="row",
    )


def test_errores_semanales_solo_en_pares_completos():
    e = errores_semanales(_pronosticos(), horizonte=2).sort("modelo", "origen", "store_id")
    # T2 en O2 no es completo; los demás: Σŷ − Σy
    assert e.select("modelo", "origen", "store_id", "y", "error").rows() == [
        ("A", O1, "T1", 30.0, 2.0),
        ("A", O1, "T2", 10.0, -2.0),
        ("A", O2, "T1", 16.0, 4.0),
        ("B", O1, "T1", 30.0, 5.0),
        ("B", O1, "T2", 10.0, 2.0),
        ("B", O2, "T1", 16.0, -4.0),
    ]


def test_perdida_por_corte_suma_los_errores_absolutos_del_corte():
    d = perdida_por_corte(errores_semanales(_pronosticos(), horizonte=2), "A", "B")
    # O1: (|2| + |−2|) − (|5| + |2|) = −3; O2: |4| − |−4| = 0. Σy de los mismos pares: 40 y 16
    assert d.rows() == [(O1, -3.0, 40.0), (O2, 0.0, 16.0)]
    # La diferencia de WAPE semanal sale de las mismas sumas: (8 − 11) / 56
    assert d.select(pl.col("d").sum() / pl.col("y").sum()).item() == pytest.approx(-3 / 56)
