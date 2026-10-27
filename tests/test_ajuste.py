"""El ajuste de LightGBM: no toca días del backtest y registra la configuración completa."""

import json
from dataclasses import asdict
from datetime import date, timedelta

import pytest

from demanda.backtest.protocolo import Protocolo
from demanda.modelos import ajuste, gbm


def test_los_origenes_de_ajuste_no_tocan_el_backtest():
    primero = date(2023, 11, 2)
    # El último horizonte puede terminar en el primer origen del backtest, no después
    ajuste.antes_del_backtest([primero - timedelta(days=7)], primero, 7)
    for origen in (primero, primero - timedelta(days=1), primero + timedelta(days=7)):
        with pytest.raises(ValueError, match="días del backtest"):
            ajuste.antes_del_backtest([primero - timedelta(days=14), origen], primero, 7)


def test_los_ajustados_salen_de_la_rejilla_y_corren_con_los_mismos_parametros():
    assert gbm.AJUSTADOS in ajuste.REJILLA
    del_backtest = gbm.LightGBMCuantiles(0.53).parametros_lgb(0.5)
    assert ajuste.parametros_registrados(gbm.AJUSTADOS) == {
        **{k: v for k, v in del_backtest.items() if k != "num_threads"},
        "num_boost_round": gbm.AJUSTADOS.n_estimators,
    }


# Dos configuraciones baratas: la prueba revisa la salida, no la calidad
RAPIDA = (
    gbm.Hiperparametros(num_leaves=7, learning_rate=0.1, n_estimators=20),
    gbm.Hiperparametros(num_leaves=15, learning_rate=0.1, n_estimators=10),
)


def test_una_corrida_diminuta_escribe_el_json(panel_sintetico, tmp_path):
    panel_path, reporte_path = tmp_path / "panel.parquet", tmp_path / "reporte.json"
    panel_sintetico.write_parquet(panel_path)
    reporte_path.write_text(json.dumps({"fuga_replenishment_signal": {"factor": 0.53}}))
    salida = tmp_path / "modelos" / "ajuste_lightgbm.json"

    r = ajuste.ejecutar(
        panel_path,
        reporte_path,
        salida,
        f"sqlite:///{tmp_path / 'mlflow.db'}",
        tmp_path / "mlruns",
        protocolo=Protocolo(n_cortes=2),
        n_origenes=2,
        rejilla=RAPIDA,
    )

    assert json.loads(salida.read_text()) == r
    # Con 2 cortes, el backtest empieza el 2023-02-25: el ajuste usa las dos semanas anteriores
    assert r["protocolo"]["primer_origen_backtest"] == "2023-02-25"
    assert r["protocolo"]["origenes"] == ["2023-02-11", "2023-02-18"]
    assert [c["nombre"] for c in r["rejilla"]] == [ajuste.nombre(h) for h in RAPIDA]
    for config, h in zip(r["rejilla"], RAPIDA, strict=True):
        assert config["parametros_lgb"] == ajuste.parametros_registrados(h)
        assert config["parametros_lgb"]["subsample_freq"] == 1
        assert config["parametros_lgb"]["deterministic"] and config["parametros_lgb"]["force_col_wise"]
        assert set(config["por_origen"]) == set(r["protocolo"]["origenes"])
    assert r["ganadora"] == min(r["rejilla"], key=lambda c: c["wape_semanal_pct"])
    assert r["referencia"]["modelo"] == "sistema_actual"
    assert {k: r["ganadora"][k] for k in asdict(gbm.AJUSTADOS)} in [asdict(h) for h in RAPIDA]
