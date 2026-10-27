"""El tablero (E8) con `AppTest`, sobre artefactos sintéticos: cada página carga, muestra sus límites y las
cifras de los artefactos, y sin artefactos dice qué comando falta. Nunca lee `artifacts/` ni el `.env`
reales."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import demanda.app
from demanda.app import datos
from demanda.cli import TABLERO, main
from demanda.config import Config

PRINCIPAL = Path(demanda.app.__file__).parent / "principal.py"
TITULOS = {
    "resumen": "Pronóstico de demanda para reposición",
    "datos": "Datos",
    "modelos": "Modelos",
    "pronostico": "Pronóstico",
    "decision": "Decisión y drift",
    "promociones": "Promociones",
    "arquitectura": "Arquitectura",
}
# Cada página con los comandos que debe pedir si no hay ningún artefacto
FALTANTES = {
    "resumen": ("make seleccion", "make decision", "make causal", "make insights"),
    "datos": ("make calidad", "make eda", "make insights"),
    "modelos": ("make backtest", "make seleccion", "make insights"),
    "pronostico": ("make backtest", "make seleccion", "make calidad"),
    "decision": ("make decision", "make drift", "make insights"),
    "promociones": ("make causal", "make insights"),
    "arquitectura": ("archify",),
}
BUEN_FIN = "El Buen Fin ocurre una sola vez"
# Los límites que la spec exige mostrar como tales, con el elemento donde van. En Pronóstico, el corte que
# abre la página (el último) tiene el Buen Fin.
LIMITES = {
    "resumen": [("caption", "costos de mantener supuestos")],
    "datos": [("info", BUEN_FIN)],
    "modelos": [("info", BUEN_FIN)],
    "pronostico": [("info", BUEN_FIN)],
    "decision": [
        ("info", "Los costos son supuestos"),
        ("info", BUEN_FIN),
        ("info", "las alertas de nivel de noviembre y enero"),
        ("info", "confunde la estacionalidad anual con drift"),
    ],
    "promociones": [("info", "Los descuentos y los márgenes son supuestos")],
}


@pytest.fixture
def abrir(monkeypatch):
    """Abre el tablero con la config apuntando a `artefactos`, `datos` y `archify`; con `pagina`, en esa
    página (si no, en la portada)."""

    def abrir_(artefactos: Path, datos_dir: Path, archify: Path, pagina: str | None = None) -> AppTest:
        monkeypatch.setenv("ARTIFACTS_DIR", str(artefactos))
        monkeypatch.setenv("DATA_DIR", str(datos_dir))
        monkeypatch.setenv("ARCHIFY_DIR", str(archify))
        st.cache_data.clear()
        at = AppTest.from_file(str(PRINCIPAL), default_timeout=60).run()
        return at if pagina is None else at.switch_page(f"paginas/{pagina}.py").run()

    return abrir_


@pytest.fixture
def editados(artefactos_app, tmp_path):
    """Una copia de los artefactos sintéticos con un JSON editado: `editados(relativa, cambio)`."""

    def editar(relativa: str, cambio: Callable[[dict], None]) -> tuple[Path, Path, Path]:
        copia = tmp_path / "artifacts"
        if not copia.exists():
            shutil.copytree(artefactos_app.artefactos, copia)
        ruta = copia / relativa
        contenido = json.loads(ruta.read_text())
        cambio(contenido)
        ruta.write_text(json.dumps(contenido))
        return copia, artefactos_app.datos, artefactos_app.archify

    return editar


def _avisos_de_faltante(at: AppTest) -> list[str]:
    return [w.value for w in at.warning if w.value.startswith("Falta")]


def _vacio(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "artifacts", tmp_path / "data", tmp_path / "archify"


def _widget(widgets, etiqueta: str):
    return next(w for w in widgets if w.label == etiqueta)


def _tabla(at: AppTest, expander: str):
    """La primera tabla dentro del desplegable con esa etiqueta, como DataFrame de pandas."""
    return next(e for e in at.expander if e.label == expander).dataframe[0].value


def _cortes(tabla) -> set[str]:
    return {o.date().isoformat() for o in tabla["origen"]}


def test_la_app_lee_el_env_de_demanda_env_file(monkeypatch, tmp_path):
    # La app crea su config igual que la CLI: el suite la apunta a un `.env` que no existe, nunca al real
    otro = tmp_path / "otro.env"
    otro.write_text(f"ARTIFACTS_DIR={tmp_path / 'desde_el_env'}\n")
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("DEMANDA_ENV_FILE", str(otro))
    assert datos.config().artifacts_dir == tmp_path / "desde_el_env"


# --- carga, avisos y límites ---------------------------------------------------------------------------


@pytest.mark.parametrize("pagina", FALTANTES)
def test_cada_pagina_carga_con_artefactos_sinteticos(pagina, artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina=pagina)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    assert at.title[0].value == TITULOS[pagina]


@pytest.mark.parametrize(("pagina", "comandos"), FALTANTES.items())
def test_sin_artefactos_cada_pagina_dice_que_comando_falta(pagina, comandos, tmp_path, abrir):
    at = abrir(*_vacio(tmp_path), pagina=pagina)
    assert not at.exception, [e.value for e in at.exception]
    avisos = " ".join(_avisos_de_faltante(at))
    for comando in comandos:
        assert f"`{comando}`" in avisos, (comando, avisos)


@pytest.mark.parametrize(("pagina", "limites"), LIMITES.items())
def test_cada_pagina_muestra_sus_limites(pagina, limites, artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina=pagina)
    for elemento, texto in limites:
        assert any(texto in e.value for e in getattr(at, elemento)), (elemento, texto)


@pytest.mark.parametrize("pagina", ["datos", "modelos", "decision", "promociones"])
def test_los_insights_de_cada_pagina_dicen_su_origen(pagina, artefactos_app, abrir):
    # Los sintéticos salen de las plantillas: la etiqueta no puede decir «caché»
    at = abrir(*artefactos_app, pagina=pagina)
    (bloque,) = [e for e in at.expander if e.label.startswith("Lo que interpreta")]
    assert " · plantilla · validador ok" in bloque.label
    assert ":orange-badge[origen: plantilla]" in [m.value for m in bloque.markdown]


# --- Resumen -------------------------------------------------------------------------------------------


def test_la_portada_es_el_resumen_con_las_cifras_de_los_artefactos(artefactos_app, abrir):
    at = abrir(*artefactos_app)
    assert not at.exception
    assert at.title[0].value == TITULOS["resumen"]
    # WAPE del campeón y del retador, ahorro del newsvendor y efecto de las promociones, sin recalcular
    assert [m.value for m in at.metric] == ["27.29 %", "20.34 %", "39.6 %", "+0.05 %"]
    # El delta del retador es la diferencia de la Diebold-Mariano, con su p
    assert at.metric[1].delta == "-6.96 pp contra el campeón"
    assert any("p = 0.166" in c.value for c in at.caption)
    # Los insights del orquestador, con su origen y el resultado del validador
    assert "Lo que concluye el orquestador" in [s.value for s in at.subheader]
    assert ":orange-badge[origen: plantilla]" in [m.value for m in at.markdown]
    assert any("ninguna rechazada" in s.value for s in at.success)


def test_el_resumen_encuentra_la_dm_en_el_otro_orden(editados, abrir):
    def invertir(s: dict) -> None:
        p = s["pruebas_dm"][0]
        p["a"], p["b"], p["diferencia_wape_pp"] = p["b"], p["a"], -p["diferencia_wape_pp"]

    at = abrir(*editados("modelos/seleccion.json", invertir))
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[1].delta == "-6.96 pp contra el campeón"


def test_el_resumen_sin_dm_lo_dice(editados, abrir):
    at = abrir(*editados("modelos/seleccion.json", lambda s: s.update(pruebas_dm=[])))
    assert not at.exception, [e.value for e in at.exception]
    assert not at.metric[1].delta
    assert any("no trae su Diebold-Mariano" in c.value for c in at.caption)


def test_el_resumen_avisa_si_falta_la_prueba_de_costo(editados, abrir):
    def quitar(d: dict) -> None:
        d["pruebas_costo"] = [p for p in d["pruebas_costo"] if not p["b"].startswith("puntual_")]

    at = abrir(*editados("decision/impacto.json", quitar))
    assert not at.exception, [e.value for e in at.exception]
    assert any("no trae la prueba de costo" in w.value for w in at.warning)
    assert "Ahorro del newsvendor" not in [m.label for m in at.metric]


# --- Modelos y Decisión --------------------------------------------------------------------------------


def test_la_tabla_de_modelos_trae_papel_y_metricas_de_los_artefactos(artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina="modelos")
    tabla = at.dataframe[0].value.set_index("modelo")
    assert tabla.loc["sistema_actual", "papel"] == "campeón"
    assert tabla.loc["lightgbm", "papel"] == "retador"
    # Las de metricas.json en el global semanal: WAPE y sesgo por separado
    assert tabla.loc["sistema_actual", "WAPE semanal (%)"] == pytest.approx(27.2929)
    assert tabla.loc["sistema_actual", "sesgo semanal (%)"] == pytest.approx(1.6794)
    assert tabla.loc["lightgbm", "sesgo semanal (%)"] == pytest.approx(-3.5995)


def test_decision_lista_las_semanas_con_inesperadas_desde_el_resumen_del_monitor(artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina="decision")
    semanas = _tabla(at, "Semanas con alertas inesperadas y sus recomendaciones")
    assert semanas["semana (origen T)"].tolist() == ["2023-11-09"]
    assert semanas["inesperadas"].tolist() == [1]


# --- Pronóstico ----------------------------------------------------------------------------------------


def test_el_pronostico_muestra_solo_la_serie_elegida(artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina="pronostico")
    _widget(at.selectbox, "Tienda").select_index(1).run()
    _widget(at.selectbox, "Categoría").set_value("Bebidas").run()
    _widget(at.select_slider, "Corte (origen T)").set_value("2023-11-02").run()
    assert not at.exception, [e.value for e in at.exception]
    tabla = _tabla(at, "La serie en tabla")
    # 1 corte × 7 días × 2 modelos (campeón y retador)
    assert len(tabla) == 14
    assert set(tabla["store_id"]) == {"STR_002"}
    assert set(tabla["category"]) == {"Bebidas"}
    assert _cortes(tabla) == {"2023-11-02"}
    assert set(tabla["modelo"]) == {"sistema_actual", "lightgbm"}
    # Ese corte es `resto`: sin la advertencia del Buen Fin
    assert not any(BUEN_FIN in i.value for i in at.info)


def test_los_cortes_seguidos_traen_todos_los_cortes(artefactos_app, abrir):
    at = abrir(*artefactos_app, pagina="pronostico")
    _widget(at.radio, "Cortes").set_value("todos").run()
    tabla = _tabla(at, "La serie en tabla")
    assert len(tabla) == 2 * 7 * 2
    assert _cortes(tabla) == {"2023-11-02", "2023-11-09"}


@pytest.mark.parametrize(
    ("filtro", "columna"), [("Formato", "store_format"), ("Región", "region"), ("NSE", "socioeconomic_level")]
)
def test_los_filtros_reducen_las_tiendas(filtro, columna, artefactos_app, abrir):
    panel = pl.read_parquet(artefactos_app.datos / "interim" / "panel.parquet")
    valor = panel.filter(pl.col("store_id") == "STR_002")[columna][0]
    assert panel.filter(pl.col("store_id") == "STR_001")[columna][0] != valor
    at = abrir(*artefactos_app, pagina="pronostico")
    _widget(at.multiselect, filtro).set_value([valor]).run()
    assert [o.split(" · ")[0] for o in _widget(at.selectbox, "Tienda").options] == ["STR_002"]


def test_tiendas_lee_solo_las_columnas_de_tienda_del_panel(artefactos_app, monkeypatch):
    ruta = artefactos_app.datos / "interim" / "panel.parquet"
    assert {"units_sold", "total_transactions", "date"} <= set(pl.read_parquet_schema(ruta))
    monkeypatch.setenv("DATA_DIR", str(artefactos_app.datos))
    st.cache_data.clear()
    tiendas = datos.tiendas()
    assert tiendas is not None
    assert tiendas.columns == list(datos.TIENDA)
    assert tiendas["store_id"].to_list() == ["STR_001", "STR_002"]


def test_el_pronostico_sin_panel_elige_la_tienda_por_su_id(artefactos_app, abrir, tmp_path):
    at = abrir(artefactos_app.artefactos, tmp_path / "data", artefactos_app.archify, pagina="pronostico")
    assert not at.exception, [e.value for e in at.exception]
    assert "`make calidad`" in " ".join(_avisos_de_faltante(at))
    assert _widget(at.selectbox, "Tienda").options == ["STR_001", "STR_002"]


# --- Arquitectura y CLI --------------------------------------------------------------------------------


def test_el_diagrama_es_el_de_la_carpeta_mas_reciente_por_su_sello(artefactos_app, tmp_path):
    carpetas = [c for c in artefactos_app.archify.iterdir() if any(c.glob("*.html"))]
    # La más vieja por sello es la última modificada: elegir por fecha de modificación daría esa
    assert max(carpetas, key=lambda c: c.stat().st_mtime).name == "architecture-solucion-20260101-120000"
    elegido = datos.diagrama_mas_reciente(artefactos_app.archify)
    assert elegido is not None
    assert elegido.parent.name == "architecture-solucion-20260201-120000"
    assert datos.diagrama_mas_reciente(tmp_path / "no_existe") is None


def test_el_flujo_marca_lo_que_falta(artefactos_app, tmp_path):
    cfg = Config(
        _env_file=None, artifacts_dir=artefactos_app.artefactos, data_dir=tmp_path, reports_dir=tmp_path
    )
    estados = {f["comando"]: f["estado"] for f in datos.estado_flujo(cfg)}
    assert estados["make seleccion"] == "listo"
    assert estados["make calidad"] == "falta"  # el reporte existe, pero no el panel de `data_dir`
    assert estados["make setup"] == "—"


def test_cli_app_lanza_streamlit_sin_navegador_ni_telemetria(monkeypatch):
    llamadas = []

    class Resultado:
        returncode = 0

    def correr(comando, check):
        llamadas.append(comando)
        return Resultado()

    monkeypatch.setattr("demanda.cli.subprocess.run", correr)
    main(["app", "--port", "8600", "--host", "0.0.0.0"])
    (comando,) = llamadas
    assert comando[1:5] == ["-m", "streamlit", "run", str(TABLERO)]
    assert PRINCIPAL.resolve() == TABLERO
    opciones = dict(zip(comando[5::2], comando[6::2], strict=True))
    assert opciones == {
        "--server.port": "8600",
        "--server.address": "0.0.0.0",
        "--server.headless": "true",
        "--browser.gatherUsageStats": "false",
        "--client.toolbarMode": "minimal",
    }
