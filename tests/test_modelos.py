from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from demanda.backtest.bucle import validar_salida
from demanda.backtest.protocolo import separar, ultima_fecha
from demanda.datos.calidad import SENAL
from demanda.datos.contratos import SERIE
from demanda.modelos import REGISTRO, SALIDA, Contexto, crear
from demanda.modelos.baselines import MediaMovil7, NaiveEstacional, SistemaActual
from demanda.modelos.ets import AutoETSModelo, rellenar_para_ajuste

FACTOR = 0.53
CONTEXTO = Contexto(factor_senal=FACTOR, n_jobs=1)
LLAVE_FUTURO = [*SERIE, "date", "h"]


def _origen(p: pl.DataFrame) -> date:
    return ultima_fecha(p) - timedelta(days=14)


def _pronosticar(nombre: str, p: pl.DataFrame, origen: date) -> pl.DataFrame:
    corte = separar(p, origen, 7)
    return crear(CONTEXTO, [nombre])[nombre].pronosticar(corte.historia, corte.futuro).sort(LLAVE_FUTURO)


@pytest.mark.parametrize("nombre", list(REGISTRO))
def test_ningun_modelo_usa_informacion_futura(panel_sintetico, alterador, nombre):
    origen = _origen(panel_sintetico)
    alterado = alterador(panel_sintetico, origen)
    # La alteración sí toca lo que no se conoce: ventas futuras y la señal del origen
    cambia = panel_sintetico.join(alterado, on=["date", *SERIE], suffix="_alt").filter(
        (pl.col("date") > origen) & (pl.col("units_sold") != pl.col("units_sold_alt"))
    )
    assert cambia.height > 0
    assert not panel_sintetico.filter(pl.col("date") == origen)[SENAL].equals(
        alterado.filter(pl.col("date") == origen)[SENAL]
    )

    assert_frame_equal(_pronosticar(nombre, panel_sintetico, origen), _pronosticar(nombre, alterado, origen))


def test_el_registro_tiene_los_baselines_en_orden():
    assert list(REGISTRO)[:4] == ["sistema_actual", "naive_estacional", "ma7", "autoets"]
    assert list(crear(CONTEXTO, ["ma7", "sistema_actual"])) == ["sistema_actual", "ma7"]
    with pytest.raises(ValueError, match="no registrados"):
        crear(CONTEXTO, ["prophet"])


def test_el_bucle_rechaza_salidas_incompletas(panel_sintetico):
    corte = separar(panel_sintetico, _origen(panel_sintetico), 7)
    pred = MediaMovil7().pronosticar(corte.historia, corte.futuro)

    assert validar_salida(pred, corte.futuro, "ma7").columns == [*SALIDA, "p10", "p90"]
    with pytest.raises(ValueError, match="una fila por fila"):
        validar_salida(pred.head(5), corte.futuro, "ma7")
    with pytest.raises(ValueError, match="una fila por fila"):
        validar_salida(pl.concat([pred.head(1), pred.slice(2)]).vstack(pred.head(1)), corte.futuro, "ma7")
    with pytest.raises(ValueError, match="faltan columnas"):
        validar_salida(pred.drop("yhat"), corte.futuro, "ma7")
    # Un nulo sacaría ese punto de la evaluación de un solo modelo
    primera = pl.int_range(pl.len()) == 0
    for malo in (None, float("nan"), float("inf")):
        roto = pred.with_columns(
            pl.when(primera).then(pl.lit(malo, pl.Float64)).otherwise("yhat").alias("yhat")
        )
        with pytest.raises(ValueError, match="no finitos"):
            validar_salida(roto, corte.futuro, "ma7")
    medio_intervalo = pred.with_columns(pl.when(primera).then(1.0).alias("p10"), pl.lit(2.0).alias("p90"))
    with pytest.raises(ValueError, match="`p10`"):
        validar_salida(medio_intervalo, corte.futuro, "ma7")


def test_sistema_actual_es_la_senal_de_ayer_entre_el_factor(panel_sintetico):
    origen = _origen(panel_sintetico)
    corte = separar(panel_sintetico, origen, 7)
    pred = SistemaActual(FACTOR).pronosticar(corte.historia, corte.futuro)
    ayer = panel_sintetico.filter(pl.col("date") == origen - timedelta(days=1)).select(
        *SERIE, (pl.col(SENAL) / FACTOR).alias("esperado")
    )
    unido = pred.join(ayer, on=SERIE)

    assert unido.height == corte.futuro.height
    assert unido["yhat"].null_count() == 0
    assert (unido["yhat"] - unido["esperado"]).abs().max() == pytest.approx(0)


def test_naive_usa_la_semana_anterior_y_sus_respaldos(panel_sintetico):
    origen = _origen(panel_sintetico)
    serie = (pl.col("store_id") == "STR_001") & (pl.col("category") == "Abarrotes")
    dia = (pl.col("date") - pl.col("date").min()).dt.total_days()

    def hace(n: int) -> date:
        return origen - timedelta(days=n)

    # y = 100 + número de día en la serie; faltan y(T−4) (para h=3) y y(T−2) e y(T−9) (para h=5)
    huecos = [hace(4), hace(2), hace(9)]
    p = panel_sintetico.with_columns(
        pl.when(serie & pl.col("date").is_in(huecos))
        .then(None)
        .when(serie)
        .then(100 + dia)
        .otherwise(pl.col("units_sold"))
        .alias("units_sold")
    )
    corte = separar(p, origen, 7)
    pred = NaiveEstacional().pronosticar(corte.historia, corte.futuro).filter(serie).sort("h")

    def y(d: date) -> float:
        return float(100 + (d - p["date"].min()).days)

    ma7 = sum(y(hace(n)) for n in range(7) if hace(n) not in huecos) / 5
    yhat = dict(zip(pred["h"], pred["yhat"], strict=True))
    assert yhat[1] == y(hace(6))  # y(T + 1 − 7)
    assert yhat[3] == y(hace(11))  # falta y(T − 4): usa y(T − 11)
    assert yhat[5] == pytest.approx(ma7)  # faltan y(T − 2) e y(T − 9): usa la MA7
    assert yhat[7] == y(origen)


def test_autoets_da_intervalo_ordenado(panel_sintetico):
    corte = separar(panel_sintetico, _origen(panel_sintetico), 7)
    pred = AutoETSModelo(n_jobs=1).pronosticar(corte.historia, corte.futuro)

    assert pred.height == corte.futuro.height
    assert pred.select("yhat", "p10", "p90").null_count().row(0) == (0, 0, 0)
    assert pred.filter((pl.col("p10") > pl.col("yhat")) | (pl.col("yhat") > pl.col("p90"))).is_empty()
    assert pred.filter(pl.col("p10") < 0).is_empty()


def test_relleno_para_ajustar_usa_la_semana_anterior_y_luego_el_mas_cercano():
    unidades: list[int | None] = [10 * (i + 1) for i in range(12)]  # el día i vale 10(i + 1)
    for i in (3, 4, 8, 10, 11):
        unidades[i] = None
    historia = pl.DataFrame(
        {
            "store_id": "STR_001",
            "category": "Bebidas",
            "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(12)],
            "units_sold": unidades,
        }
    )
    relleno = rellenar_para_ajuste(historia)["y"].to_list()

    assert relleno[8] == 20  # la semana anterior (día 1)
    assert (relleno[3], relleno[4]) == (30, 60)  # sin semana anterior: el día observado más cercano
    assert (relleno[10], relleno[11]) == (100, 100)  # su semana anterior también falta: el día 9
    assert relleno[:3] == [10, 20, 30]
