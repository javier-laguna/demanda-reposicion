"""El protocolo real del proceso hijo de Chronos-2, con un `chronos` falso puesto en `PYTHONPATH`.

El hijo es el de producción (`python -P -m demanda.proceso_chronos`); solo cambia el paquete que importa. Así
corre en CI sin torch ni pesos. Cada llamada puede pedir un comportamiento con el argumento `simular`.
"""

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from demanda.backtest.bucle import validar_salida
from demanda.backtest.protocolo import separar, ultima_fecha
from demanda.modelos import chronos2
from demanda.proceso_chronos import PipelineAislado

STUB = """
import os
import time

import pandas as pd


class _Pipeline:
    def predict_df(self, df, future_df=None, simular="nada", **kwargs):
        if simular == "ruido":
            print("ruido de print en stdout")
            os.write(1, b"ruido crudo en el descriptor 1\\n")
        elif simular == "error":
            raise ValueError("falla simulada del pipeline")
        elif simular == "muere":
            os._exit(3)
        elif simular == "lento":
            time.sleep(60)
        medias = df.groupby("item_id")["target"].mean()
        salida = future_df[["item_id", "timestamp"]].copy()
        nivel = salida["item_id"].map(medias)
        for q in kwargs["quantile_levels"]:
            salida[str(q)] = nivel * (0.5 + q)
        return salida


class Chronos2Pipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if os.environ.get("STUB_CHRONOS_NO_CARGA"):
            raise OSError("pesos no encontrados (simulado)")
        print("ruido en stdout al cargar")
        return _Pipeline()
"""


@pytest.fixture(scope="module")
def stub(tmp_path_factory) -> Path:
    carpeta = tmp_path_factory.mktemp("stub")
    (carpeta / "chronos").mkdir()
    (carpeta / "chronos" / "__init__.py").write_text(STUB)
    return carpeta


@pytest.fixture(scope="module")
def con_stub(stub):
    """El `PYTHONPATH` que hereda el hijo, con el paquete falso primero."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(stub), os.environ.get("PYTHONPATH")) if p))
        yield mp


@pytest.fixture(scope="module")
def hijo(con_stub):
    pipeline = PipelineAislado()
    yield pipeline
    pipeline.cerrar()


@pytest.fixture(scope="module")
def corte(panel_sintetico):
    return separar(panel_sintetico, ultima_fecha(panel_sintetico) - timedelta(days=14), 7)


def _entrada() -> tuple[pd.DataFrame, pd.DataFrame]:
    fechas = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"item_id": ["a"] * 5 + ["b"] * 5, "timestamp": list(fechas[:5]) * 2})
    df["target"] = [1.0, 2.0, 3.0, float("nan"), 4.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    futuro = pd.DataFrame({"item_id": ["a", "a", "b", "b"], "timestamp": list(fechas[5:7]) * 2})
    return df, futuro


def test_ida_y_vuelta_del_pronostico(hijo):
    df, futuro = _entrada()
    salida = hijo.predict_df(df, future_df=futuro, quantile_levels=[0.1, 0.5, 0.9])
    # Media sin el nulo: 2.5 en a y 10 en b; el cuantil q vale media × (0.5 + q)
    assert salida["0.5"].tolist() == [2.5, 2.5, 10.0, 10.0]
    assert salida["0.9"].tolist() == pytest.approx([3.5, 3.5, 14.0, 14.0])
    assert list(salida["timestamp"]) == list(futuro["timestamp"])


def test_el_ruido_en_stdout_no_rompe_el_protocolo(hijo):
    df, futuro = _entrada()
    ruidosa = hijo.predict_df(df, future_df=futuro, quantile_levels=[0.5], simular="ruido")
    assert ruidosa["0.5"].tolist() == [2.5, 2.5, 10.0, 10.0]


def test_un_error_del_hijo_llega_claro_y_el_hijo_sigue(hijo):
    df, futuro = _entrada()
    with pytest.raises(RuntimeError, match="falla simulada del pipeline"):
        hijo.predict_df(df, future_df=futuro, quantile_levels=[0.5], simular="error")
    assert hijo.predict_df(df, future_df=futuro, quantile_levels=[0.5])["0.5"].tolist()[0] == 2.5


def test_el_modelo_usa_el_hijo_de_punta_a_punta(con_stub, corte):
    modelo = chronos2.Chronos2(cargador=chronos2.cargar_pipeline)
    try:
        pred = modelo.pronosticar(corte.historia, corte.futuro)
    finally:
        modelo.cerrar()
    assert validar_salida(pred, corte.futuro, "chronos2").height == corte.futuro.height
    assert pred.filter((pl.col("p10") > pl.col("yhat")) | (pl.col("yhat") > pl.col("p90"))).is_empty()


def test_la_muerte_del_hijo_da_un_mensaje_claro(con_stub):
    df, futuro = _entrada()
    pipeline = PipelineAislado()
    with pytest.raises(RuntimeError, match=r"terminó.*código 3"):
        pipeline.predict_df(df, future_df=futuro, quantile_levels=[0.5], simular="muere")
    # Ya muerto, la siguiente llamada falla igual de claro (el pipe está roto)
    with pytest.raises(RuntimeError, match="terminó"):
        pipeline.predict_df(df, future_df=futuro, quantile_levels=[0.5])
    pipeline.cerrar()


def test_si_el_hijo_no_responde_a_tiempo_se_termina(con_stub):
    df, futuro = _entrada()
    pipeline = PipelineAislado(timeout=1)
    with pytest.raises(RuntimeError, match="no respondió en 1 s"):
        pipeline.predict_df(df, future_df=futuro, quantile_levels=[0.5], simular="lento")
    assert not pipeline.activo()


def test_una_falla_al_cargar_sale_al_crear(con_stub):
    con_stub.setenv("STUB_CHRONOS_NO_CARGA", "1")
    try:
        with pytest.raises(RuntimeError, match="pesos no encontrados"):
            PipelineAislado()
    finally:
        con_stub.delenv("STUB_CHRONOS_NO_CARGA")


def test_cerrar_termina_al_hijo(con_stub):
    with PipelineAislado() as pipeline:
        assert pipeline.activo()
    assert not pipeline.activo()
