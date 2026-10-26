"""El backtest de punta a punta y la conciliación con E2, sobre datos sintéticos."""

import json
from datetime import date

import polars as pl
import pytest

from demanda.backtest import bucle, conciliacion, etapa, segmentos, seguimiento
from demanda.backtest.metricas import escala_mase
from demanda.backtest.protocolo import Protocolo, separar
from demanda.datos import calidad, panel
from demanda.datos.contratos import SERIE
from demanda.modelos import REGISTRO
from demanda.modelos.baselines import MediaMovil7, SistemaActual


def test_backtest_de_punta_a_punta(generar_tablas, carpeta_csv, tmp_path):
    raw = carpeta_csv(generar_tablas(dias=70))
    panel_path, reporte_path = tmp_path / "panel.parquet", tmp_path / "reporte.json"
    calidad.ejecutar(raw, panel_path, reporte_path)
    salida, mlruns = tmp_path / "backtest", tmp_path / "mlruns"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    argumentos = {
        "panel_path": panel_path,
        "reporte_path": reporte_path,
        "resumen_path": tmp_path / "no_existe.json",
        "salida_dir": salida,
        "tracking_uri": uri,
        "mlruns_dir": mlruns,
        "protocolo": Protocolo(n_cortes=2),
        "n_jobs": 1,
    }

    primera = etapa.ejecutar(**argumentos)
    escrito = (salida / "metricas.json").read_text()
    etapa.ejecutar(**argumentos)
    assert (salida / "metricas.json").read_text() == escrito

    m = json.loads(escrito)
    assert m == json.loads(json.dumps(primera.metricas))
    assert {"fuente", "protocolo", "cobertura", "conciliacion_e2", "segmentos", "definiciones"} <= set(m)
    assert m["protocolo"]["modelos"] == list(REGISTRO)
    # Sin resumen del EDA la referencia queda en nulo, sin fallar
    assert m["conciliacion_e2"]["referencia_e2_pct"] is None
    assert m["conciliacion_e2"]["diferencia_pp"] is None
    semanal = m["cobertura"]["semanal"]
    assert semanal["pares_total"] == 2 * 6 and 0 < semanal["pares_completos"] <= semanal["pares_total"]

    pronosticos = pl.read_parquet(salida / "pronosticos.parquet")
    assert tuple(pronosticos.columns) == bucle.COLUMNAS
    assert pronosticos.height == len(REGISTRO) * 2 * 6 * 7

    tabla = pl.DataFrame(m["segmentos"])
    globales = tabla.filter(pl.col("segmento") == "global")
    assert set(globales["modelo"]) == set(REGISTRO)
    assert (tabla["n"] <= tabla["n_total"]).all()
    # El pinball medio y la cobertura solo existen para el modelo con intervalo
    con_intervalo = tabla.filter(pl.col("metrica") == "cobertura_80_pct")["modelo"].unique().to_list()
    assert con_intervalo == ["autoets"]

    from mlflow import MlflowClient

    cliente = MlflowClient(tracking_uri=uri)
    experimento = cliente.get_experiment_by_name(seguimiento.EXPERIMENTO)
    assert experimento is not None
    assert experimento.artifact_location == mlruns.resolve().as_uri()
    runs = cliente.search_runs([experimento.experiment_id])
    assert len(runs) == 2 * len(REGISTRO)
    run = runs[0]
    assert run.data.params["n_cortes"] == "2"
    assert run.data.metrics["semanal_wape_pct"] > 0
    assert [a.path for a in cliente.list_artifacts(run.info.run_id)] == ["metricas.json"]


def test_la_escala_del_mase_se_mide_en_la_historia_de_cada_corte(panel_sintetico):
    protocolo = Protocolo(n_cortes=2)
    escalas = bucle.correr(panel_sintetico, {"ma7": MediaMovil7()}, protocolo).escalas
    for origen in protocolo.origenes(panel_sintetico["date"].max()):
        hasta_t = escala_mase(panel_sintetico.filter(pl.col("date") <= origen))
        del_corte = escalas.filter(pl.col("origen") == origen).drop("origen")
        unido = del_corte.join(hasta_t, on=SERIE, suffix="_esperada")
        assert unido.height == panel_sintetico.select(SERIE).n_unique()
        assert (unido["escala"] - unido["escala_esperada"]).abs().max() == pytest.approx(0)
    # Con el panel completo cambiaría: la escala no puede ver el horizonte
    completa = escala_mase(panel_sintetico)
    assert (
        not escalas.filter(pl.col("origen") == origen).drop("origen").sort(SERIE).equals(completa.sort(SERIE))
    )


def test_conciliacion_usa_la_definicion_del_backtest(generar_tablas):
    # 240 días para que haya orígenes desde 2023-07-01
    p = panel.construir(generar_tablas(dias=240)).sort([*SERIE, "date"])
    factor, origen = 0.53, date(2023, 7, 15)
    diarios = conciliacion.origenes_diarios(p, factor, 7)

    corte = separar(p, origen, 7)
    semanal = (
        SistemaActual(factor)
        .pronosticar(corte.historia, corte.futuro)
        .group_by(SERIE)
        .agg(pl.col("yhat").sum())
    )
    reales = (
        p.filter(pl.col("date").is_between(date(2023, 7, 16), date(2023, 7, 22)))
        .group_by(SERIE)
        .agg(pl.col("units_sold").sum().cast(pl.Float64).alias("real"))
    )
    unido = (
        diarios.filter(pl.col("origen") == origen)
        .join(semanal, on=SERIE, suffix="_backtest")
        .join(reales, on=SERIE)
    )
    assert unido.height == p.select(SERIE).n_unique()
    assert (unido["yhat"] - unido["yhat_backtest"]).abs().max() == pytest.approx(0)
    completos = unido.filter("completo")
    assert (completos["y"] == completos["real"]).all()

    resumen = {
        "sistema_actual": {
            "wape_pct": {"senal_T_menos_1_factor_ajustado": 12.0},
            "origenes": {"evaluados": 5},
        }
    }
    r = conciliacion.conciliar(p, factor, 7, resumen)
    assert diarios["origen"].min() == conciliacion.INICIO_E2
    assert 0 < r["origenes_evaluados"] <= r["origenes_posibles"] == diarios.height
    assert r["diferencia_pp"] == pytest.approx(r["wape_pct"] - 12.0, abs=1e-3)
    assert r["origenes_evaluados_e2"] == 5
    assert conciliacion.conciliar(p, factor, 7, None)["referencia_e2_pct"] is None


def test_la_semana_posterior_al_buen_fin_no_cuenta_como_normal(panel_sintetico):
    p = panel_sintetico.with_columns(
        pl.col("date").is_in([date(2023, 2, 10), date(2023, 2, 11)]).alias("is_buen_fin")
    )
    origenes = pl.Series([date(2023, 2, 9), date(2023, 2, 16), date(2023, 2, 17), date(2023, 2, 23)])

    post = dict(segmentos.posterior_al_buen_fin(origenes, p).iter_rows())
    assert post == {
        date(2023, 2, 9): False,  # el Buen Fin cae en su horizonte, no en su historia
        date(2023, 2, 16): True,
        date(2023, 2, 17): True,
        date(2023, 2, 23): False,
    }
    casos = pl.DataFrame(
        {
            "bf": [True, False, False, False],
            "nav": [True, True, False, False],
            "post": [True, True, True, False],
        }
    )
    niveles = casos.select(segmentos.evento(pl.col("bf"), pl.col("nav"), pl.col("post")))
    assert niveles.to_series().to_list() == ["buen_fin", "navidad", "post_buen_fin", "resto"]
