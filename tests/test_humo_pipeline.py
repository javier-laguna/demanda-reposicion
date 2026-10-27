"""Prueba de humo de `demanda pipeline` y las reglas del comando.

La de humo corre las etapas de calidad a insights por la CLI sobre un conjunto sintético de 40 tiendas × 6
categorías × 425 días, con el calendario del real. Todo se escribe en un temporal: `DATA_DIR`,
`ARTIFACTS_DIR`, `REPORTS_DIR` e `INSIGHTS_DIR` (la caché de insights) apuntan ahí, y el `.env` es el
inexistente del conftest.

- **40 tiendas, no 4:** la Wald agrupada por tienda del balance de E6 (B1) prueba 36 pendientes y necesita
  más clústeres que restricciones; con 4 tiendas `causal` falla por diseño.
- El backtest y la calibración usan solo `sistema_actual,lightgbm`: Chronos-2 bajaría pesos.
- LightGBM entrena con `ARBOLES` árboles en lugar de los de `AJUSTADOS`: es una prueba de que las etapas
  se conectan, no del modelo, y con los 600 árboles solo el backtest tardaría ~6 min.
"""

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from demanda import cli
from demanda.agentes.modelos import ORDEN
from demanda.config import RAIZ, Config, env_file, get_config
from demanda.datos.contratos import CATEGORIAS, Tablas
from demanda.datos.fuente import ARCHIVOS
from demanda.modelos import gbm
from demanda.modelos.registro import REGISTRO
from demanda.proceso_chronos import SinPesos

INICIO, FIN = date(2023, 1, 1), date(2024, 2, 29)
N_TIENDAS = 40
ARBOLES = 50
FESTIVOS = {
    date(2023, 1, 1): "Año Nuevo",
    date(2023, 2, 6): "Día de la Constitución",
    date(2023, 3, 20): "Natalicio Benito Juárez",
    date(2023, 4, 6): "Jueves Santo",
    date(2023, 4, 7): "Viernes Santo",
    date(2023, 5, 1): "Día del Trabajo",
    date(2023, 9, 16): "Día de la Independencia",
    date(2023, 11, 2): "Día de Muertos",
    date(2023, 11, 17): "Buen Fin",
    date(2023, 11, 18): "Buen Fin",
    date(2023, 11, 19): "Buen Fin",
    date(2023, 11, 20): "Revolución Mexicana / Buen Fin",
    date(2023, 12, 12): "Virgen de Guadalupe",
    date(2023, 12, 24): "Nochebuena",
    date(2023, 12, 25): "Navidad",
    date(2023, 12, 31): "Fin de Año",
    date(2024, 1, 1): "Año Nuevo",
    date(2024, 2, 5): "Día de la Constitución",
}
TEMPORADA = {12: "Invierno", 1: "Invierno", 2: "Invierno", 3: "Primavera", 4: "Primavera", 5: "Primavera"}
TEMPORADA |= {6: "Verano", 7: "Verano", 8: "Verano", 9: "Otoño", 10: "Otoño", 11: "Otoño"}
# Filas que faltan en la malla, como los huecos de dos días del real
HUECO = ("STR_002", (date(2023, 2, 1), date(2023, 2, 2)))

ARTEFACTOS = (
    "calidad/reporte.json",
    "eda/resumen.json",
    "backtest/pronosticos.parquet",
    "backtest/metricas.json",
    "modelos/seleccion.json",
    "backtest/calibracion.parquet",
    "backtest/calibracion.json",
    "decision/impacto.json",
    "drift/monitor.json",
    "causal/promociones.json",
    "insights/insights.json",
    "mlflow.db",
)


def _entre(desde: date, hasta: date) -> pl.Expr:
    return pl.col("date").is_between(desde, hasta)


def calendario(fechas: pl.Series) -> pl.DataFrame:
    """El calendario del real: festivos, quincenas (el 15 y el último día del mes), estaciones, temporada
    navideña (del 15 de diciembre al 6 de enero, y los primeros 6 días de 2023), Buen Fin y Semana Santa."""
    return pl.DataFrame({"date": fechas}).with_columns(
        (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("day_of_week"),
        pl.col("date").dt.strftime("%A").alias("day_name"),
        pl.col("date").dt.week().cast(pl.Int64).alias("week_of_year"),
        pl.col("date").dt.month().cast(pl.Int64).alias("month"),
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
        pl.col("date").dt.quarter().cast(pl.Int64).alias("quarter"),
        pl.col("date").dt.month().replace_strict(TEMPORADA, return_dtype=pl.String).alias("season"),
        pl.col("date").is_in(list(FESTIVOS)).alias("is_holiday"),
        pl.col("date").replace_strict(FESTIVOS, default=None, return_dtype=pl.String).alias("holiday_name"),
        ((pl.col("date").dt.day() == 15) | (pl.col("date") == pl.col("date").dt.month_end())).alias(
            "is_payday"
        ),
        (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
        (_entre(date(2023, 1, 1), date(2023, 1, 6)) | _entre(date(2023, 12, 15), date(2024, 1, 6))).alias(
            "is_navidad_season"
        ),
        _entre(date(2023, 11, 17), date(2023, 11, 20)).alias("is_buen_fin"),
        _entre(date(2023, 4, 6), date(2023, 4, 7)).alias("is_semana_santa"),
    )


def tablas_humo(generar) -> Tablas:
    """Las tablas de `generar` (todos los formatos, regiones y NSE) con el calendario del real y un hueco en
    la malla."""
    t = generar(n_tiendas=N_TIENDAS, categorias=CATEGORIAS, dias=(FIN - INICIO).days + 1, semilla=9)
    tienda, (desde, hasta) = HUECO
    transacciones = t.transacciones.filter(~((pl.col("store_id") == tienda) & _entre(desde, hasta)))
    return Tablas(transacciones, t.tiendas, calendario(t.calendario["date"]))


def colocar(tablas: Tablas, data_dir: Path) -> None:
    """Lo que hace `make data`: los CSV en `data/raw/` y sus hashes en `data/manifest.sha256`."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    tablas.transacciones.write_csv(raw / "transactions.csv")
    tablas.tiendas.write_csv(raw / "stores.csv")
    tablas.calendario.write_csv(raw / "calendar.csv")
    lineas = [f"{hashlib.sha256((raw / n).read_bytes()).hexdigest()}  {n}" for n in ARCHIVOS]
    (data_dir / "manifest.sha256").write_text("\n".join(lineas) + "\n")


def arbol(carpeta: Path) -> dict[str, tuple[int, int]]:
    """Tamaño y fecha de modificación de cada archivo bajo `carpeta`, para ver que nada cambió."""
    return {
        str(r.relative_to(carpeta)): (r.stat().st_size, r.stat().st_mtime_ns)
        for r in carpeta.rglob("*")
        if r.is_file()
    }


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Las rutas de la config en el temporal; la config se vuelve a leer antes y después de la prueba.

    `DEMANDA_MLRUNS_DIR` va fuera de `ARTIFACTS_DIR`, para ver que el backtest la sigue. Con
    `PYTHONPYCACHEPREFIX`, los procesos hijos escriben `__pycache__` y la caché de sesión de marimo
    (`__marimo__`) en el temporal; `demanda eda` además descarta la suya."""
    rutas = {
        "DATA_DIR": tmp_path / "data",
        "ARTIFACTS_DIR": tmp_path / "artifacts",
        "REPORTS_DIR": tmp_path / "reports",
        "INSIGHTS_DIR": tmp_path / "insights",
        "DEMANDA_MLRUNS_DIR": tmp_path / "runs",
        "PYTHONPYCACHEPREFIX": tmp_path / "pycache",
    }
    for variable, ruta in rutas.items():
        monkeypatch.setenv(variable, str(ruta))
    for variable in ("DEMANDA_MLFLOW_TRACKING_URI", "LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    get_config.cache_clear()
    yield rutas
    get_config.cache_clear()


@pytest.fixture
def lightgbm_ligero(monkeypatch):
    def fabrica(ctx):
        hiper = replace(gbm.AJUSTADOS, n_estimators=ARBOLES)
        return gbm.LightGBMCuantiles(ctx.factor_senal, hiper=hiper, semilla=ctx.semilla)

    monkeypatch.setitem(REGISTRO, "lightgbm", fabrica)


def test_el_pipeline_corre_de_calidad_a_insights(generar_tablas, entorno, lightgbm_ligero, capsys):
    assert not env_file().exists()  # el conftest apunta DEMANDA_ENV_FILE a un archivo inexistente
    colocar(tablas_humo(generar_tablas), entorno["DATA_DIR"])
    notebooks = arbol(RAIZ / "notebooks")
    cli.main(["pipeline", "--modelos", "sistema_actual,lightgbm"])
    salida = capsys.readouterr().out
    assert arbol(RAIZ / "notebooks") == notebooks  # ni el EDA ni marimo escriben en el repo

    artefactos = entorno["ARTIFACTS_DIR"]
    for relativa in ARTEFACTOS:
        assert (artefactos / relativa).exists(), relativa
    assert (entorno["DATA_DIR"] / "interim" / "panel.parquet").exists()
    assert (entorno["REPORTS_DIR"] / "eda.html").exists()
    assert any(entorno["DEMANDA_MLRUNS_DIR"].iterdir())
    assert not (artefactos / "mlruns").exists()

    metricas = json.loads((artefactos / "backtest" / "metricas.json").read_text())
    assert metricas["protocolo"]["modelos"] == ["sistema_actual", "lightgbm"]
    assert len(metricas["protocolo"]["origenes"]) == 17
    # Sin key, `auto` cae a la caché; en el temporal no hay caché, así que los cuatro dominios son plantilla
    insights = json.loads((artefactos / "insights" / "insights.json").read_text())
    assert (insights["modo_solicitado"], insights["modo"]) == ("auto", "cache")
    assert list(insights["dominios"]) == list(ORDEN)
    assert all(d["origen"] == "plantilla" for d in insights["dominios"].values())
    assert insights["validacion"]["todas_validas"]
    assert not entorno["INSIGHTS_DIR"].exists()  # sin red, la caché no se escribe

    tiempos = salida.split("tiempo por etapa:")[-1]
    for etapa in cli.ETAPAS:
        assert f"== {etapa} ==" in salida
        assert f"  {etapa:<12}" in tiempos


# --- reglas del comando, sin correr etapas ---------------------------------------------------------------


def test_desde_retoma_en_la_etapa_pedida():
    assert cli.etapas_a_correr() == list(cli.ETAPAS)
    assert cli.etapas_a_correr("drift") == ["drift", "causal", "insights"]
    with pytest.raises(ValueError, match="etapa desconocida"):
        cli.etapas_a_correr("ajustar")


def _config_temporal(monkeypatch, tmp_path: Path) -> Config:
    cfg = Config(_env_file=None, data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    return cfg


def test_sin_los_csv_pide_make_data_y_lista_los_que_faltan(tmp_path, monkeypatch, capsys):
    cfg = _config_temporal(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["pipeline"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "make data" in err and "calendar.csv, stores.csv, transactions.csv" in err

    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "stores.csv").touch()
    with pytest.raises(SystemExit):
        cli.main(["pipeline"])
    assert "faltan calendar.csv, transactions.csv en" in capsys.readouterr().err


def _falla(error: BaseException):
    def etapa(args):
        raise error

    return etapa


@pytest.mark.parametrize(
    ("error", "mensaje"),
    [
        (FileNotFoundError("no existe la calibración"), "error: no existe la calibración"),
        (SinPesos("no se pudieron cargar los pesos"), "error: no se pudieron cargar los pesos"),
        # Como el EDA: marimo ya imprimió su error
        (SystemExit(1), None),
        (RuntimeError("inesperado"), "Traceback"),
    ],
)
def test_se_detiene_en_la_primera_etapa_que_falla(error, mensaje, tmp_path, monkeypatch, capsys):
    cfg = _config_temporal(monkeypatch, tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    for nombre in ARCHIVOS:
        (cfg.raw_dir / nombre).touch()
    corridas = []
    for nombre in cli.ETAPAS:
        monkeypatch.setitem(cli.ETAPAS, nombre, lambda args, nombre=nombre: corridas.append(nombre))
    monkeypatch.setitem(cli.ETAPAS, "decision", _falla(error))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["pipeline", "--desde", "seleccion"])
    assert excinfo.value.code == 1
    assert corridas == ["seleccion", "calibracion"]
    salida = capsys.readouterr()
    assert "retoma con `demanda pipeline --desde decision`" in salida.err
    assert "tiempo por etapa:" in salida.out and "decision" in salida.out and "(falló)" in salida.out
    if mensaje is None:
        assert "error:" not in salida.err
    else:
        assert mensaje in salida.err
    if mensaje != "Traceback":
        assert "Traceback" not in salida.err  # solo lo inesperado lleva traceback


def test_sin_pesos_de_chronos_el_comando_falla_sin_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_backtest", _falla(SinPesos("no se pudieron cargar los pesos de Chronos-2")))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["backtest"])
    assert excinfo.value.code == 1
    assert capsys.readouterr().err == "error: no se pudieron cargar los pesos de Chronos-2\n"
