from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_series_equal

from demanda.backtest.protocolo import separar
from demanda.datos.contratos import SERIE
from demanda.features import matriz
from demanda.modelos.baselines import MediaMovil7, SistemaActual

FACTOR = 0.53


@pytest.fixture(scope="module")
def origen(panel_sintetico):
    return panel_sintetico["date"].max() - timedelta(days=14)


@pytest.fixture(scope="module")
def matrices(panel_sintetico, alterador, origen):
    """La matriz con el panel original y con el futuro de `origen` alterado; incluye un origen anterior."""
    origenes = [origen - timedelta(days=7), origen]
    original = matriz.construir(panel_sintetico, origenes, 7, FACTOR)
    alterada = matriz.construir(alterador(panel_sintetico, origen), origenes, 7, FACTOR)
    return original, alterada


@pytest.mark.parametrize("columna", matriz.FEATURES)
def test_ninguna_feature_usa_informacion_futura(matrices, columna):
    original, alterada = matrices
    assert_series_equal(original[columna], alterada[columna])


def test_la_alteracion_si_cambia_el_objetivo(matrices):
    original, alterada = matrices
    assert not original[matriz.OBJETIVO].equals(alterada[matriz.OBJETIVO])


def test_forma_de_la_matriz(matrices, panel_sintetico):
    original, _ = matrices
    series = panel_sintetico.select(SERIE).n_unique()

    assert original.height == 2 * 7 * series
    assert original.select("origen", *SERIE, "h").is_duplicated().sum() == 0
    assert original.filter(pl.col("date") != pl.col("origen") + pl.duration(days=pl.col("h"))).is_empty()
    with pytest.raises(ValueError, match="después del origen"):
        matriz.construir(panel_sintetico, [original["origen"].max()], 8, FACTOR)


def test_features_del_origen_coinciden_con_los_baselines(matrices, panel_sintetico, origen):
    original, _ = matrices
    del_origen = original.filter(pl.col("origen") == origen)
    corte = separar(panel_sintetico, origen, 7)
    llave = [*SERIE, "date"]
    sistema = SistemaActual(FACTOR).pronosticar(corte.historia, corte.futuro).select(*llave, "yhat")
    ma7 = MediaMovil7().pronosticar(corte.historia, corte.futuro).select(*llave, pl.col("yhat").alias("ma7"))
    unido = del_origen.join(sistema, on=llave).join(ma7, on=llave)

    assert unido.height == del_origen.height
    assert_series_equal(unido["senal_previa"], unido["yhat"], check_names=False)
    assert_series_equal(unido["media_7"], unido["ma7"], check_names=False)
    # Con h = 7, el rezago de una semana es el día del origen
    h7 = del_origen.filter(pl.col("h") == 7)
    assert_series_equal(h7["y_d_menos_7"], h7["y_origen"], check_names=False)
