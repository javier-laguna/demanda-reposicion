from datetime import date, timedelta

import polars as pl

from demanda.backtest.protocolo import CONOCIDAS, DE_LA_SENAL, Protocolo, separar
from demanda.datos.calidad import SENAL
from demanda.datos.contratos import SERIE


def test_origenes_son_17_jueves_sin_traslape():
    ultima = date(2024, 2, 29)  # la última fecha de los datos reales
    origenes = Protocolo().origenes(ultima)

    assert len(origenes) == 17
    assert all(o.weekday() == 3 for o in origenes)
    assert (origenes[0], origenes[-1]) == (date(2023, 11, 2), date(2024, 2, 22))
    dias = [o + timedelta(days=h) for o in origenes for h in range(1, 8)]
    # Sin traslape ni huecos: del día siguiente al primer origen hasta la última fecha
    assert dias == [date(2023, 11, 3) + timedelta(days=i) for i in range(len(dias))]
    assert dias[-1] == ultima
    # El Buen Fin (2023-11-17 a 20) cae completo en el horizonte del corte del 2023-11-16
    assert date(2023, 11, 16) in origenes


def test_lo_conocido_de_antemano_es_una_lista_cerrada():
    # Explícita a propósito: la prueba de fugas altera todo lo que no está en CONOCIDAS, así que una columna
    # agregada aquí por error escaparía a esa prueba
    tiendas = ("store_format", "region", "size_sqm", "num_checkouts", "opening_year", "socioeconomic_level")
    calendario = ("day_of_week", "day_name", "week_of_year", "month", "year", "quarter", "season")
    eventos = ("is_holiday", "holiday_name", "is_payday", "is_weekend", "is_navidad_season", "is_buen_fin")
    assert set(CONOCIDAS) == {
        "date",
        "store_id",
        "category",
        *tiendas,
        "has_pharmacy",
        "has_fuel_station",
        *calendario,
        *eventos,
        "is_semana_santa",
        "has_promotion",
    }


def test_separar_no_entrega_nada_del_futuro(panel_sintetico):
    origen = panel_sintetico["date"].max() - timedelta(days=7)
    corte = separar(panel_sintetico, origen, 7)
    series = panel_sintetico.select(SERIE).n_unique()

    assert corte.historia["date"].max() == origen
    en_origen = corte.historia.filter(pl.col("date") == origen)
    assert all(en_origen[c].null_count() == series for c in DE_LA_SENAL)
    antes = panel_sintetico.filter(pl.col("date") < origen)[SENAL]
    assert corte.historia.filter(pl.col("date") < origen)[SENAL].equals(antes)

    assert set(corte.futuro.columns) == {*CONOCIDAS, "h"}
    assert "units_sold" not in corte.futuro.columns and SENAL not in corte.futuro.columns
    assert corte.futuro["h"].unique().sort().to_list() == list(range(1, 8))
    assert corte.futuro.height == 7 * series
    assert corte.futuro.filter(pl.col("date") != pl.lit(origen) + pl.duration(days=pl.col("h"))).is_empty()
