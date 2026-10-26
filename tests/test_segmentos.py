"""Métricas por segmento contra valores calculados a mano.

Son 2 series × 2 cortes de 2 días, con un hueco y un Buen Fin.
"""

from datetime import date

import polars as pl
import pytest

from demanda.backtest import segmentos
from demanda.backtest.bucle import COLUMNAS, Resultado
from demanda.backtest.protocolo import Protocolo

O1, O2 = date(2023, 1, 10), date(2023, 1, 12)
D11, D12, D13, D14 = (date(2023, 1, d) for d in (11, 12, 13, 14))

# (origen, tienda, fecha, h, y, ŷ). El Buen Fin es el 11: O1 lo tiene en su horizonte y O2 en sus
# días previos.
# Error ŷ − y: +2, −5, 0, (sin real), −2, 0, +10, 0
PUNTOS = [
    (O1, "T1", D11, 1, 10, 12),
    (O1, "T1", D12, 2, 20, 15),
    (O1, "T2", D11, 1, 30, 30),
    (O1, "T2", D12, 2, None, 25),
    (O2, "T1", D13, 1, 10, 8),
    (O2, "T1", D14, 2, 10, 10),
    (O2, "T2", D13, 1, 40, 50),
    (O2, "T2", D14, 2, 20, 20),
]
# Distinta por corte en T1, para que unirla sin el origen cambie el MASE
ESCALAS = {(O1, "T1"): 2.0, (O2, "T1"): 4.0, (O1, "T2"): 5.0, (O2, "T2"): 5.0}


def _resultado() -> Resultado:
    filas = []
    for modelo, ancho in (("puntual", None), ("con_intervalo", 3.0)):
        for origen, tienda, d, h, y, yhat in PUNTOS:
            p10, p90 = (None, None) if ancho is None else (yhat - ancho, yhat + ancho)
            filas.append((modelo, origen, tienda, "Abarrotes", d, h, y, yhat, p10, p90))
    tipos = [pl.String, pl.Date, pl.String, pl.String, pl.Date, pl.Int64, *[pl.Float64] * 4]
    pronosticos = pl.DataFrame(filas, schema=dict(zip(COLUMNAS, tipos, strict=True)), orient="row")
    escalas = pl.DataFrame(
        [(o, t, "Abarrotes", e) for (o, t), e in ESCALAS.items()],
        schema={"origen": pl.Date, "store_id": pl.String, "category": pl.String, "escala": pl.Float64},
        orient="row",
    )
    return Resultado(pronosticos, escalas)


def _panel() -> pl.DataFrame:
    tiendas = {"T1": ("Bodega", "Centro", "A/B"), "T2": ("Express", "Norte", "C")}
    return pl.DataFrame(
        [(d, t, "Abarrotes", *tiendas[t], d == D11, False) for d in (D11, D12, D13, D14) for t in tiendas],
        schema=[
            ("date", pl.Date),
            ("store_id", pl.String),
            ("category", pl.String),
            ("store_format", pl.String),
            ("region", pl.String),
            ("socioeconomic_level", pl.String),
            ("is_buen_fin", pl.Boolean),
            ("is_navidad_season", pl.Boolean),
        ],
        orient="row",
    )


@pytest.fixture(scope="module")
def tabla() -> pl.DataFrame:
    protocolo = Protocolo(n_cortes=2, horizonte=2)
    return segmentos.tabla(_resultado(), _panel(), protocolo, ["puntual", "con_intervalo"])


def _fila(tabla: pl.DataFrame, granularidad: str, segmento: str, nivel: str, metrica: str, modelo="puntual"):
    f = tabla.filter(
        (pl.col("granularidad") == granularidad)
        & (pl.col("segmento") == segmento)
        & (pl.col("nivel") == nivel)
        & (pl.col("modelo") == modelo)
        & (pl.col("metrica") == metrica)
    )
    assert f.height == 1, (granularidad, segmento, nivel, metrica, modelo)
    return f.row(0, named=True)


@pytest.mark.parametrize(
    ("segmento", "nivel", "wape", "n", "n_total"),
    [
        # Pares completos: T1·O1 (30 vs 27), T1·O2 (20 vs 18), T2·O2 (60 vs 70); T2·O1 tiene un hueco
        ("global", "global", 100 * 15 / 110, 3, 4),
        ("evento", "buen_fin", 100 * 3 / 30, 1, 2),
        ("evento", "post_buen_fin", 100 * 12 / 80, 2, 2),
        ("formato", "Bodega", 100 * 5 / 50, 2, 2),
        ("formato", "Express", 100 * 10 / 60, 1, 2),
        ("corte", str(O1), 100 * 3 / 30, 1, 2),
    ],
)
def test_wape_semanal_por_segmento(tabla, segmento, nivel, wape, n, n_total):
    fila = _fila(tabla, "semanal", segmento, nivel, "wape_pct")
    assert fila["valor"] == pytest.approx(wape, abs=1e-4)
    assert (fila["n"], fila["n_total"]) == (n, n_total)


def test_sesgo_semanal(tabla):
    assert _fila(tabla, "semanal", "global", "global", "sesgo_pct")["valor"] == pytest.approx(
        100 * 5 / 110, abs=1e-4
    )
    assert _fila(tabla, "semanal", "evento", "post_buen_fin", "sesgo_pct")["valor"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("segmento", "nivel", "wape", "n", "n_total"),
    [
        ("global", "global", 100 * 19 / 140, 7, 8),
        ("evento", "buen_fin", 100 * 2 / 40, 2, 2),  # el día 11
        ("evento", "resto", 100 * 5 / 20, 1, 2),  # el 12: O1 no tiene el Buen Fin en sus días previos
        ("evento", "post_buen_fin", 100 * 12 / 80, 4, 4),
        ("horizonte", "1", 100 * 14 / 90, 4, 4),
        ("horizonte", "2", 100 * 5 / 50, 3, 4),
    ],
)
def test_wape_diario_por_segmento(tabla, segmento, nivel, wape, n, n_total):
    fila = _fila(tabla, "diaria", segmento, nivel, "wape_pct")
    assert fila["valor"] == pytest.approx(wape, abs=1e-4)
    assert (fila["n"], fila["n_total"]) == (n, n_total)


def test_mase_usa_la_escala_de_cada_corte(tabla):
    # MAE / escala por serie y corte: T1·O1 3.5/2, T2·O1 0/5, T1·O2 1/4, T2·O2 5/5
    esperado = (1.75 + 0 + 0.25 + 1.0) / 4
    assert _fila(tabla, "diaria", "global", "global", "mase")["valor"] == pytest.approx(esperado, abs=1e-4)


def test_pinball_y_cobertura(tabla):
    def valor(metrica: str, modelo: str = "con_intervalo") -> float:
        return _fila(tabla, "diaria", "global", "global", metrica, modelo)["valor"]

    # q = ŷ ∓ 3. Pérdidas por día: τ = 0.1 suma 8.6, τ = 0.5 suma 9.5 (½ Σ|e|), τ = 0.9 suma 4.6; Σy = 140
    assert valor("pinball_10_pct") == pytest.approx(100 * 8.6 / 140, abs=1e-4)
    assert valor("pinball_50_pct") == pytest.approx(100 * 9.5 / 140, abs=1e-4)
    assert valor("pinball_90_pct") == pytest.approx(100 * 4.6 / 140, abs=1e-4)
    assert valor("pinball_medio_pct") == pytest.approx(100 * (8.6 + 9.5 + 4.6) / 3 / 140, abs=1e-4)
    # Dentro de [ŷ − 3, ŷ + 3]: los 5 días con |e| ≤ 3 de 7
    assert valor("cobertura_80_pct") == pytest.approx(100 * 5 / 7, abs=1e-4)
    # El modelo puntual solo tiene pinball_50
    assert valor("pinball_50_pct", "puntual") == pytest.approx(100 * 9.5 / 140, abs=1e-4)
    puntual = tabla.filter((pl.col("modelo") == "puntual") & pl.col("metrica").str.starts_with("cobertura"))
    assert puntual.is_empty()
