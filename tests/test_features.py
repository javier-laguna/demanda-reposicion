from datetime import date, timedelta

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


def test_features_de_evento_y_mediana_calculadas_a_mano(panel_sintetico):
    """Una serie con 10 unidades en los 28 días hasta T, salvo un pico de 1000 en T y un nulo en T − 3."""
    t = date(2023, 2, 15)

    def hace(n: int) -> date:
        return t - timedelta(days=n)

    serie = (pl.col("store_id") == "STR_001") & (pl.col("category") == "Abarrotes")
    buen_fin, festivos = [hace(1), hace(2), hace(15)], [hace(5), hace(30)]
    p = panel_sintetico.with_columns(
        pl.when(serie & (pl.col("date") == t))
        .then(1000)
        .when(serie & (pl.col("date") == hace(3)))
        .then(None)
        .when(serie & pl.col("date").is_between(hace(27), t))
        .then(10)
        .otherwise(pl.col("units_sold"))
        .alias("units_sold"),
        pl.col("date").is_in(buen_fin).alias("is_buen_fin"),
        pl.col("date").is_in(festivos).alias("is_holiday"),
    )
    m = matriz.construir(p, [t], 7, FACTOR).filter(serie).sort("h")

    # 26 días con 10 y uno con 1000: la mediana no se mueve, la media sí
    assert m["mediana_28"].unique().to_list() == [10.0]
    assert m["media_28"][0] == pytest.approx((26 * 10 + 1000) / 27)
    # Buen Fin en T − 1 y T − 2 (en los 7 días) y en T − 15 (solo en los 28); festivo en T − 5, no en T − 30
    assert (m["dias_buen_fin_7"][0], m["dias_buen_fin_28"][0], m["dias_festivo_7"][0]) == (2, 3, 1)
    # El día del rezago es d − k = T + h − k
    assert m["evento_d_menos_7"].to_list() == [False, True, False, False, True, True, False]  # T−5, T−2, T−1
    assert not m["evento_d_menos_14"].any()  # de T − 13 a T − 7
    assert m["evento_d_menos_21"].to_list() == [False] * 5 + [True, False]  # T − 15 con h = 6
    assert not m["evento_d_menos_28"].any()  # de T − 27 a T − 21


def test_las_ventanas_de_evento_son_por_fecha_y_no_por_serie(panel_sintetico):
    origen = panel_sintetico["date"].max() - timedelta(days=7)
    p = panel_sintetico.with_columns((pl.col("date") == origen).alias("is_buen_fin"))
    m = matriz.construir(p, [origen], 7, FACTOR)
    assert m["dias_buen_fin_7"].unique().to_list() == [1]
    # Con h = 7, el rezago de una semana es el origen
    assert m.filter(pl.col("h") == 7)["evento_d_menos_7"].all()


@pytest.mark.parametrize(
    ("buen_fin", "festivos", "esperado"),
    [
        # Justo fuera: T + 1 (el calendario se conoce, pero la ventana termina en T), T − 7 para las de 7
        # días y T − 28 para la de 28. Solo T − 7 cae en la de 28
        ([1, -7, -28], [1, -7], (0, 1, 0)),
        # Justo dentro: T, T − 6 y T − 27
        ([0, -6, -27], [0, -6], (2, 3, 2)),
    ],
    ids=["justo fuera", "justo dentro"],
)
def test_las_ventanas_de_evento_terminan_en_el_origen(panel_sintetico, buen_fin, festivos, esperado):
    """Una ventana corrida un día a cualquier lado, o un día más larga o más corta, cambia algún conteo."""
    t = date(2023, 2, 15)

    def dias(desfases: list[int]) -> list[date]:
        return [t + timedelta(days=n) for n in desfases]

    p = panel_sintetico.with_columns(
        pl.col("date").is_in(dias(buen_fin)).alias("is_buen_fin"),
        pl.col("date").is_in(dias(festivos)).alias("is_holiday"),
    )
    m = matriz.construir(p, [t], 7, FACTOR)
    conteos = m.select("dias_buen_fin_7", "dias_buen_fin_28", "dias_festivo_7").unique()
    assert conteos.rows() == [esperado]
