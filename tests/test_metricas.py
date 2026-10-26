"""Métricas contra ejemplos calculados a mano."""

from datetime import date, timedelta

import polars as pl
import pytest

from demanda.backtest import metricas


@pytest.fixture
def puntos() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [10.0, 20.0, 30.0, 40.0],
            "yhat": [12.0, 18.0, 33.0, 40.0],
            "p10": [8.0, 25.0, 20.0, 35.0],
            "p90": [15.0, 30.0, 40.0, 45.0],
        }
    )


def _valor(df: pl.DataFrame, expr: pl.Expr) -> float:
    return df.select(expr).item()


def test_wape_y_sesgo(puntos):
    # |e| = 2 + 2 + 3 + 0 = 7 y e = 2 − 2 + 3 + 0 = 3, sobre Σy = 100
    assert _valor(puntos, metricas.wape()) == pytest.approx(0.07)
    assert _valor(puntos, metricas.sesgo()) == pytest.approx(0.03)


def test_pinball(puntos):
    # En la mediana es la mitad del WAPE
    assert _valor(puntos, metricas.pinball(0.5, "yhat")) == pytest.approx(0.035)
    # u = y − p10 = 2, −5, 10, 5 → 0.1·2 + 0.9·5 + 0.1·10 + 0.1·5 = 6.2
    assert _valor(puntos, metricas.pinball(0.1, "p10")) == pytest.approx(6.2 / 100)
    # u = y − p90 = −5, −10, −10, −5 → 0.1 · 30 = 3
    assert _valor(puntos, metricas.pinball(0.9, "p90")) == pytest.approx(3 / 100)


def test_cobertura(puntos):
    # 10 ∈ [8, 15], 20 ∉ [25, 30], 30 ∈ [20, 40], 40 ∈ [35, 45]
    assert _valor(puntos, metricas.cobertura("p10", "p90")) == pytest.approx(0.75)


def test_columna_de_cada_cuantil():
    assert [metricas.columna_cuantil(t) for t in (0.1, 0.5, 0.9)] == ["p10", "yhat", "p90"]


def test_escala_mase_es_el_mae_del_naive_estacional():
    fechas = [date(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    historia = pl.DataFrame(
        {
            "store_id": "STR_001",
            "category": "Bebidas",
            "date": fechas,
            # y(t) − y(t − 7) existe en los tres últimos días: |10 − 1|, |4 − 2| y el último, que es nulo
            "units_sold": [1, 2, 3, 4, 5, 6, 7, 10, 4, None],
        }
    )
    assert metricas.escala_mase(historia)["escala"].item() == pytest.approx((9 + 2) / 2)


def test_mase_promedia_series_por_corte():
    df = pl.DataFrame(
        {
            "modelo": "m",
            "origen": date(2023, 3, 1),
            "store_id": ["A", "A", "B", "C"],
            "category": "Bebidas",
            "y": [10.0, 10.0, 5.0, 3.0],
            "yhat": [11.0, 13.0, 7.0, 3.0],
            # A: MAE 2 / 4 = 0.5; B: 2 / 1 = 2; C tiene escala 0 y no entra
            "escala": [4.0, 4.0, 1.0, 0.0],
        }
    )
    assert metricas.mase(df, ["modelo"])["mase"].item() == pytest.approx(1.25)
