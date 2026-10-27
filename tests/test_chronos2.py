"""Chronos-2 con un doble del pipeline; el modelo real solo corre si sus pesos ya están en la caché local."""

from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from demanda.backtest.bucle import validar_salida
from demanda.backtest.protocolo import separar, ultima_fecha
from demanda.modelos import Contexto, chronos2, crear


@pytest.fixture(scope="module")
def corte(panel_sintetico):
    return separar(panel_sintetico, ultima_fecha(panel_sintetico) - timedelta(days=14), 7)


def test_llama_al_pipeline_con_el_contexto_completo_y_las_covariables(corte, doble_chronos):
    doble = doble_chronos()
    pred = chronos2.Chronos2(cargador=lambda: doble).pronosticar(corte.historia, corte.futuro)
    (llamada,) = doble.llamadas
    df, conocido = llamada["df"], llamada["future_df"]

    assert {k: llamada[k] for k in ("prediction_length", "quantile_levels", "freq", "cross_learning")} == {
        "prediction_length": 7,
        "quantile_levels": [0.1, 0.5, 0.9],
        "freq": "D",
        "cross_learning": False,
    }
    # Contexto: toda la historia, con los nulos como NaN; nada de la señal ni de las demás medidas
    assert list(df.columns) == ["item_id", "timestamp", "target", *chronos2.COVARIABLES]
    assert len(df) == corte.historia.height
    assert df["target"].isna().sum() == corte.historia["units_sold"].null_count() > 0
    assert df["timestamp"].max() == pd.Timestamp(ultima_fecha(corte.historia))
    # Futuro: solo las covariables conocidas, como 0/1
    assert list(conocido.columns) == ["item_id", "timestamp", *chronos2.COVARIABLES]
    assert len(conocido) == corte.futuro.height
    assert set(np.unique(conocido[list(chronos2.COVARIABLES)].to_numpy())) <= {0.0, 1.0}

    assert validar_salida(pred, corte.futuro, "chronos2").height == corte.futuro.height
    assert pred.filter((pl.col("p10") > pl.col("yhat")) | (pl.col("yhat") > pl.col("p90"))).is_empty()


def test_carga_los_pesos_una_sola_vez(corte, doble_chronos):
    cargas = []

    def cargador():
        cargas.append(1)
        return doble_chronos()

    modelo = chronos2.Chronos2(cargador=cargador)
    modelo.pronosticar(corte.historia, corte.futuro)
    modelo.pronosticar(corte.historia, corte.futuro)
    assert len(cargas) == 1


def test_ordena_y_recorta_los_cuantiles(corte, doble_chronos):
    class Cruzado(doble_chronos):
        def predict_df(self, df, future_df, **kwargs):
            salida = super().predict_df(df, future_df, **kwargs)
            return salida.assign(**{"0.1": 5.0, "0.5": -1.0, "0.9": 3.0})

    pred = chronos2.Chronos2(cargador=Cruzado).pronosticar(corte.historia, corte.futuro)
    assert pred.select("p10", "yhat", "p90").unique().rows() == [(0.0, 3.0, 5.0)]


def test_el_registro_usa_el_doble_en_las_pruebas(corte):
    modelo = crear(Contexto(factor_senal=0.53), ["chronos2"])["chronos2"]
    assert isinstance(modelo, chronos2.Chronos2)
    assert type(modelo.pipeline()).__name__ == "PipelineDoble"


def _pesos_en_cache() -> bool:
    from huggingface_hub import try_to_load_from_cache

    ruta = try_to_load_from_cache(chronos2.MODELO, "model.safetensors", revision=chronos2.REVISION)
    return isinstance(ruta, str)


CON_PESOS = pytest.mark.skipif(
    not _pesos_en_cache(), reason="los pesos de Chronos-2 no están en la caché local"
)


@pytest.fixture(scope="module")
def modelo_real():
    """El modelo real, con su proceso hijo, compartido por las pruebas de este módulo."""
    if not _pesos_en_cache():
        pytest.skip("los pesos de Chronos-2 no están en la caché local")
    modelo = chronos2.Chronos2(cargador=partial(chronos2.cargar_pipeline, solo_local=True))
    yield modelo
    modelo.cerrar()


@CON_PESOS
def test_el_modelo_real_es_determinista(modelo_real, corte):
    primero = modelo_real.pronosticar(corte.historia, corte.futuro)

    assert validar_salida(primero, corte.futuro, "chronos2").height == corte.futuro.height
    assert primero.filter((pl.col("p10") > pl.col("yhat")) | (pl.col("yhat") > pl.col("p90"))).is_empty()
    assert primero.filter(pl.col("p10") < 0).is_empty()
    assert_frame_equal(primero, modelo_real.pronosticar(corte.historia, corte.futuro))


@CON_PESOS
def test_el_modelo_real_no_usa_informacion_futura(modelo_real, panel_sintetico, alterador, corte):
    alterado = separar(alterador(panel_sintetico, corte.origen), corte.origen, 7)
    assert_frame_equal(
        modelo_real.pronosticar(corte.historia, corte.futuro),
        modelo_real.pronosticar(alterado.historia, alterado.futuro),
    )
