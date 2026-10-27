"""La calibración de E5: orígenes explícitos en el bucle, el campeón y su retador, y los artefactos."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from demanda import cli
from demanda.backtest import bucle, calibracion
from demanda.backtest.protocolo import Protocolo
from demanda.config import Config
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256
from demanda.decision import etapa as decision
from demanda.drift import etapa as drift
from demanda.modelos import Contexto, crear
from demanda.seleccion.campeon import campeon_y_retador


def test_con_los_origenes_del_protocolo_el_bucle_produce_lo_mismo(panel_sintetico):
    protocolo = Protocolo(n_cortes=2)
    modelos = crear(Contexto(factor_senal=0.53), ["sistema_actual", "ma7"])
    por_defecto = bucle.correr(panel_sintetico, modelos, protocolo)
    origenes = protocolo.origenes(panel_sintetico["date"].max())
    explicitos = bucle.correr(panel_sintetico, modelos, protocolo, origenes=origenes)
    assert explicitos.pronosticos.equals(por_defecto.pronosticos)
    # `escala_mase` agrupa sin orden: se comparan las mismas filas
    llave = ["origen", *SERIE]
    assert explicitos.escalas.sort(llave).equals(por_defecto.escalas.sort(llave))

    # Un origen suelto es el mismo corte: mismo `separar`, mismo horizonte
    solo = bucle.correr(panel_sintetico, modelos, protocolo, origenes=[origenes[1]])
    assert solo.pronosticos.equals(por_defecto.pronosticos.filter(pl.col("origen") == origenes[1]))
    # Y uno fuera de la rejilla también corre, con sus 7 días
    otro = bucle.correr(panel_sintetico, modelos, protocolo, origenes=[date(2023, 2, 20)])
    assert otro.pronosticos["date"].min() == date(2023, 2, 21)
    assert otro.pronosticos["h"].max() == 7


def test_el_retador_es_el_de_menor_wape_global_distinto_del_campeon():
    seleccion = {
        "campeon": "sistema_actual",
        "modelos": {
            "sistema_actual": {"wape_semanal_global_pct": 27.3},
            "autoets": {"wape_semanal_global_pct": 24.7},
            "lightgbm": {"wape_semanal_global_pct": 20.3},
            "chronos2": {"wape_semanal_global_pct": 20.3},  # empata: gana el primero del registro
        },
    }
    assert campeon_y_retador(seleccion) == ("sistema_actual", "lightgbm")
    seleccion["campeon"] = "lightgbm"
    assert campeon_y_retador(seleccion) == ("lightgbm", "chronos2")
    with pytest.raises(ValueError, match="retadores"):
        campeon_y_retador({"campeon": "a", "modelos": {"a": {"wape_semanal_global_pct": 1.0}}})


def test_la_calibracion_corre_al_campeon_y_al_retador_antes_del_backtest(artefactos_e5):
    a = artefactos_e5
    cal = pl.read_parquet(a.calibracion)
    campeon, retador = campeon_y_retador(json.loads(a.seleccion.read_text()))
    assert tuple(cal.columns) == bucle.COLUMNAS
    assert cal["modelo"].unique(maintain_order=True).to_list() == [
        m for m in ("sistema_actual", "ma7") if m in (campeon, retador)
    ]
    assert sorted(cal["origen"].unique().to_list()) == [date(2023, 3, 11), date(2023, 3, 18)]
    # El último horizonte termina en el primer corte del backtest (2023-03-25), no después
    assert cal["date"].max() == date(2023, 3, 25)
    assert cal.height == 2 * 2 * 10 * 6 * 7
    # El registro trae la fuente con hashes, los modelos y los orígenes
    registro = json.loads(calibracion.registro(a.calibracion).read_text())
    assert registro["fuente"] == {
        "calibracion_sha256": sha256(a.calibracion),
        "panel_sha256": sha256(a.panel),
        "reporte_calidad_sha256": sha256(a.reporte),
        "seleccion_sha256": sha256(a.seleccion),
    }
    assert registro["modelos"] == cal["modelo"].unique(maintain_order=True).to_list()
    assert registro["origenes"] == ["2023-03-11", "2023-03-18"]

    # Los reales son los del panel, y los pronósticos, los del mismo modelo con el mismo `separar`
    p = pl.read_parquet(a.panel)
    reales = cal.join(p.select(*SERIE, "date", "units_sold"), on=[*SERIE, "date"])
    assert reales.select((pl.col("y") == pl.col("units_sold")).fill_null(True).all()).item()
    modelos = crear(Contexto(factor_senal=0.53), [campeon, retador])
    directo = bucle.correr(p, modelos, Protocolo(n_cortes=4), origenes=[date(2023, 3, 11), date(2023, 3, 18)])
    assert directo.pronosticos.sort("modelo", "origen", *SERIE, "date").equals(
        cal.sort("modelo", "origen", *SERIE, "date")
    )


def test_leer_rechaza_otros_modelos_o_dias_del_backtest(artefactos_e5):
    a = artefactos_e5
    campeon, retador = campeon_y_retador(json.loads(a.seleccion.read_text()))
    fuente = (a.panel, a.seleccion)
    assert calibracion.leer(a.calibracion, *fuente, [campeon, retador], date(2023, 3, 25), 7).height > 0
    with pytest.raises(ValueError, match="corre `demanda calibracion`"):
        calibracion.leer(a.calibracion, *fuente, [campeon, "lightgbm"], date(2023, 3, 25), 7)
    with pytest.raises(ValueError, match="días del backtest"):
        calibracion.leer(a.calibracion, *fuente, [campeon, retador], date(2023, 3, 24), 7)


def _copia(a, carpeta: Path) -> dict[str, Path]:
    """Los artefactos sintéticos copiados a `carpeta`, para alterarlos sin tocar el fixture."""
    copias = {}
    for nombre in ("panel", "pronosticos", "metricas", "seleccion", "calibracion"):
        origen = getattr(a, nombre)
        copias[nombre] = carpeta / origen.name
        copias[nombre].write_bytes(origen.read_bytes())
    registro = calibracion.registro(a.calibracion)
    (carpeta / registro.name).write_bytes(registro.read_bytes())
    return copias


@pytest.mark.parametrize("cambio", ["panel", "seleccion", "calibracion", "sin_registro"])
def test_decision_y_drift_rechazan_una_calibracion_de_otra_corrida(artefactos_e5, tmp_path, cambio):
    c = _copia(artefactos_e5, tmp_path)
    if cambio == "panel":
        # Rehacer la calidad cambia el panel: la calibración ya no es la suya
        p = pl.read_parquet(c["panel"])
        p.with_columns(pl.col("units_sold").fill_null(0)).write_parquet(c["panel"])
    elif cambio == "seleccion":
        # La misma fuente de E4, pero otro archivo: `verificar_fuente` pasa y la calibración no
        s = json.loads(c["seleccion"].read_text())
        c["seleccion"].write_text(json.dumps({**s, "nota": "otra corrida"}))
    elif cambio == "calibracion":
        pl.read_parquet(c["calibracion"]).with_columns(pl.col("yhat") + 1).write_parquet(c["calibracion"])
    else:
        calibracion.registro(c["calibracion"]).unlink()
    mensaje = "no existe" if cambio == "sin_registro" else "otra corrida"
    with pytest.raises((ValueError, FileNotFoundError), match=mensaje):
        decision.ejecutar(
            c["pronosticos"], c["metricas"], c["seleccion"], c["calibracion"], c["panel"], tmp_path / "i.json"
        )
    with pytest.raises((ValueError, FileNotFoundError), match=mensaje):
        drift.ejecutar(
            c["panel"], c["pronosticos"], c["metricas"], c["seleccion"], c["calibracion"], tmp_path / "m.json"
        )
    assert not (tmp_path / "i.json").exists() and not (tmp_path / "m.json").exists()


def test_con_la_calibracion_de_su_corrida_las_dos_etapas_corren(artefactos_e5, tmp_path):
    c = _copia(artefactos_e5, tmp_path)
    decision.ejecutar(
        c["pronosticos"], c["metricas"], c["seleccion"], c["calibracion"], c["panel"], tmp_path / "i.json"
    )
    assert (tmp_path / "i.json").exists()


@pytest.mark.parametrize(
    ("comando", "falta"),
    [("calibracion", "corre antes `demanda calidad`"), ("decision", "corre antes `demanda calidad`")],
)
def test_sin_artefactos_previos_el_comando_falla_con_un_mensaje_claro(
    comando, falta, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "get_config", lambda: Config(_env_file=None, data_dir=tmp_path, artifacts_dir=tmp_path)
    )
    with pytest.raises(SystemExit):
        cli.main([comando])
    assert falta in capsys.readouterr().err


def test_sin_calibracion_decision_y_drift_piden_correrla(artefactos_e5, tmp_path, monkeypatch, capsys):
    a = artefactos_e5
    (tmp_path / "interim").mkdir()
    (tmp_path / "interim" / "panel.parquet").write_bytes(a.panel.read_bytes())
    for carpeta, archivo, origen in (
        ("calidad", "reporte.json", a.reporte),
        ("backtest", "pronosticos.parquet", a.pronosticos),
        ("backtest", "metricas.json", a.metricas),
        ("modelos", "seleccion.json", a.seleccion),
    ):
        (tmp_path / carpeta).mkdir(exist_ok=True)
        (tmp_path / carpeta / archivo).write_bytes(origen.read_bytes())
    monkeypatch.setattr(
        cli, "get_config", lambda: Config(_env_file=None, data_dir=tmp_path, artifacts_dir=tmp_path)
    )
    for comando in ("decision", "drift"):
        with pytest.raises(SystemExit):
            cli.main([comando])
        assert "corre antes `demanda calibracion`" in capsys.readouterr().err
