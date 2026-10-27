from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from demanda.backtest.bucle import validar_salida
from demanda.backtest.protocolo import separar, ultima_fecha
from demanda.datos.contratos import CATEGORIAS
from demanda.features import matriz
from demanda.modelos import gbm
from demanda.modelos.registro import ordenar_cuantiles

FACTOR = 0.53
# Pocos árboles: las pruebas revisan el contrato, no la calidad del modelo
RAPIDO = gbm.Hiperparametros(num_leaves=7, learning_rate=0.1, n_estimators=30, min_child_samples=5)


@pytest.fixture(scope="module")
def corte(panel_sintetico):
    return separar(panel_sintetico, ultima_fecha(panel_sintetico) - timedelta(days=14), 7)


def test_cuantiles_ordenados_y_no_negativos(corte):
    pred = gbm.LightGBMCuantiles(FACTOR, hiper=RAPIDO).pronosticar(corte.historia, corte.futuro)

    assert validar_salida(pred, corte.futuro, "lightgbm").height == corte.futuro.height
    assert pred.filter((pl.col("p10") > pl.col("yhat")) | (pl.col("yhat") > pl.col("p90"))).is_empty()
    assert pred.filter(pl.col("p10") < 0).is_empty()
    # El intervalo no es degenerado: los tres cuantiles salen de modelos distintos
    assert pred.select((pl.col("p90") - pl.col("p10")).mean()).item() > 0


def test_pronosticar_ordena_y_recorta_cuantiles_cruzados(corte, monkeypatch):
    """Con boosters que dan p10 > p90 y una mediana negativa, `pronosticar` debe ordenar y recortar."""

    class Falso:
        def __init__(self, valor: float) -> None:
            self.valor = valor

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(len(x), self.valor)

    falsos = {0.1: Falso(5.0), 0.5: Falso(-1.0), 0.9: Falso(3.0)}
    monkeypatch.setattr(gbm.LightGBMCuantiles, "ajustar", lambda self, *args, **kwargs: falsos)
    pred = gbm.LightGBMCuantiles(FACTOR).pronosticar(corte.historia, corte.futuro)

    # Relativos al nivel: −1 sube a 0 y queda como p10; 3 y 5 pasan a yhat y p90
    assert (pred["p10"] == 0).all()
    assert (pred["yhat"] > 0).all()
    assert (pred["p90"] / pred["yhat"] - 5 / 3).abs().max() == pytest.approx(0)


def test_dos_ajustes_dan_el_mismo_pronostico_con_cualquier_numero_de_hilos(corte):
    def pronostico(hilos: int) -> pl.DataFrame:
        modelo = gbm.LightGBMCuantiles(FACTOR, hiper=RAPIDO, hilos=hilos)
        return modelo.pronosticar(corte.historia, corte.futuro)

    primero = pronostico(1)
    assert_frame_equal(primero, pronostico(1))
    assert_frame_equal(primero, pronostico(2))


def test_origenes_de_entrenamiento_cada_tres_dias_hacia_atras():
    t = date(2023, 1, 20)
    origenes = gbm.origenes_de_entrenamiento(t, date(2023, 1, 1), 3)
    assert origenes == [date(2023, 1, d) for d in (1, 4, 7, 10, 13, 16, 19)]
    assert gbm.origenes_de_entrenamiento(t, date(2023, 1, 2), 3)[0] == date(2023, 1, 4)


def test_entrena_solo_con_objetivos_hasta_el_origen(corte):
    t = ultima_fecha(corte.historia)
    m = gbm.entrenamiento(corte.historia, 7, FACTOR, 3)

    assert m["date"].max() == t  # el día T sí entra, como h = 1 del origen T − 1
    assert m["origen"].max() == t - timedelta(days=1)
    assert m[matriz.OBJETIVO].null_count() == 0
    assert m.filter(pl.col("date") != pl.col("origen") + pl.duration(days=pl.col("h"))).is_empty()
    # Del origen T − 1 solo queda h = 1; del T − 4, h ≤ 4
    assert m.filter(pl.col("origen") == t - timedelta(days=1))["h"].unique().to_list() == [1]
    assert m.filter(pl.col("origen") == t - timedelta(days=4))["h"].max() == 4


def _matriz(**columnas) -> pl.DataFrame:
    """Dos filas de la matriz con todas las features en nulo salvo `columnas`."""
    base = {c: [None, None] for c in matriz.FEATURES}
    return pl.DataFrame({**base, **columnas}, schema_overrides={c: pl.Float64 for c in matriz.EN_UNIDADES})


def test_nivel_y_features_relativas():
    m = _matriz(
        mediana_28=[0.4, 40.0],
        media_28=[2.0, 55.0],
        y_origen=[3.0, 20.0],
        y_d_menos_7=[1.0, 80.0],
        category=["Bebidas", "Ropa"],
        holiday_name=[None, "Navidad"],
        h=[1, 7],
    )
    x, nivel = gbm.a_numpy(m)
    col = {c: i for i, c in enumerate(gbm.COLUMNAS)}

    # El nivel es la mediana, no la media; si es menor que 1, sube a 1
    assert nivel.tolist() == [1.0, 40.0]
    assert x[:, col["y_origen"]].tolist() == [3.0, 0.5]
    assert x[:, col["y_d_menos_7"]].tolist() == [1.0, 2.0]
    assert x[:, col["log_nivel"]].tolist() == pytest.approx([0.0, 3.6888794541])
    # Las categóricas usan el código de su dominio fijo, no el de los datos del corte
    assert x[:, col["category"]].tolist() == [CATEGORIAS.index("Bebidas"), CATEGORIAS.index("Ropa")]
    assert x[1, col["holiday_name"]] == gbm.FESTIVOS.index("Navidad")
    assert x[:, col["h"]].tolist() == [1.0, 7.0]


def test_sin_ventas_en_28_dias_el_nivel_es_uno():
    # La mediana y la media de 28 días son nulas a la vez: solo si no hay ningún real en la ventana
    _, nivel = gbm.a_numpy(_matriz(mediana_28=[None, 5.0]))
    assert nivel.tolist() == [1.0, 5.0]


def test_un_festivo_desconocido_no_pasa_en_silencio():
    with pytest.raises(pl.exceptions.InvalidOperationError):
        gbm.a_numpy(_matriz(holiday_name=["Día de la Marmota", None]))


def test_ordenar_cuantiles_cuando_se_cruzan():
    pred = pl.DataFrame({"p10": [5.0, 1.0], "yhat": [-1.0, 2.0], "p90": [3.0, 4.0]})
    ordenado = ordenar_cuantiles(pred)
    assert ordenado.rows() == [(0.0, 3.0, 5.0), (1.0, 2.0, 4.0)]
    # Sin intervalo, solo recorta yhat
    assert ordenar_cuantiles(pl.DataFrame({"yhat": [-2.0, 3.0]}))["yhat"].to_list() == [0.0, 3.0]


def test_los_parametros_fijan_el_determinismo():
    p = gbm.LightGBMCuantiles(FACTOR).parametros_lgb(0.9)
    assert p["objective"] == "quantile" and p["alpha"] == 0.9
    assert p["deterministic"] and p["force_col_wise"]
    assert p["subsample_freq"] == 1  # sin él, LightGBM ignora subsample
    assert "n_estimators" not in p  # va como num_boost_round
    assert "store_id" not in gbm.COLUMNAS  # la tienda entra por sus atributos
