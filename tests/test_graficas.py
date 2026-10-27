"""Las gráficas del tablero, sin Streamlit: cada función devuelve una gráfica de Altair válida con los
artefactos sintéticos, con las cifras tal cual salen de ellos, y los intervalos van como banda o barra de
error."""

import inspect
import json
from collections.abc import Callable

import altair as alt
import polars as pl
import pytest

from demanda.app import graficas


def _leer(artefactos_app, relativa: str) -> dict:
    return json.loads((artefactos_app.artefactos / relativa).read_text())


def _serie(artefactos_app) -> pl.DataFrame:
    return pl.read_parquet(artefactos_app.artefactos / "backtest" / "pronosticos.parquet").filter(
        pl.col("store_id") == "STR_001", pl.col("category") == "Bebidas"
    )


# Cada función pública de `graficas.py` con cómo llamarla sobre los artefactos sintéticos
LLAMADAS: dict[str, Callable] = {
    "nulos": lambda a: graficas.nulos(_leer(a, "calidad/reporte.json")),
    "pesos_fuga": lambda a: graficas.pesos_fuga(_leer(a, "calidad/reporte.json")),
    "efectos_calendario": lambda a: graficas.efectos_calendario(_leer(a, "eda/resumen.json")),
    "perfil_semanal": lambda a: graficas.perfil_semanal(_leer(a, "eda/resumen.json")),
    "metrica_por_segmento": lambda a: graficas.metrica_por_segmento(
        _leer(a, "backtest/metricas.json"), "evento", "sesgo_pct"
    ),
    "cobertura_intervalos": lambda a: graficas.cobertura_intervalos(_leer(a, "backtest/metricas.json")),
    "pronostico": lambda a: graficas.pronostico(_serie(a), "sistema_actual", "lightgbm"),
    "costo_por_politica": lambda a: graficas.costo_por_politica(_leer(a, "decision/impacto.json")),
    "sensibilidad_costo": lambda a: graficas.sensibilidad_costo(_leer(a, "decision/impacto.json")),
    "alertas_por_semana": lambda a: graficas.alertas_por_semana(_leer(a, "drift/monitor.json"), "calidad"),
    "vigilancia_retador": lambda a: graficas.vigilancia_retador(_leer(a, "drift/monitor.json")),
    "estimaciones": lambda a: graficas.estimaciones(_leer(a, "causal/promociones.json")["robustez"]),
    "efecto_por_categoria": lambda a: graficas.efecto_por_categoria(
        _leer(a, "causal/promociones.json"), 0.05
    ),
    "potencia_semanal": lambda a: graficas.potencia_semanal(_leer(a, "causal/promociones.json")),
}


def _marca(capa: dict) -> str:
    marca = capa["mark"]
    return marca["type"] if isinstance(marca, dict) else marca


def _marcas(especificacion: dict) -> list[str]:
    """Las marcas de una especificación de Vega-Lite, también dentro de las capas."""
    propias = [_marca(especificacion)] if "mark" in especificacion else []
    return propias + [m for capa in especificacion.get("layer", []) for m in _marcas(capa)]


def _capa(especificacion: dict, marca: str, n: int = 0) -> tuple[dict, list[dict]]:
    """La `n`-ésima capa con esa marca y sus filas."""
    capa = [c for c in especificacion["layer"] if _marca(c) == marca][n]
    return capa, especificacion["datasets"][capa["data"]["name"]]


def _esperadas(metricas: dict, segmentos: tuple[str, ...], metrica: str, granularidad: str) -> set:
    return {
        (f["nivel"], f["modelo"], f["valor"])
        for f in metricas["segmentos"]
        if f["segmento"] in segmentos and f["metrica"] == metrica and f["granularidad"] == granularidad
    }


def test_todas_las_funciones_publicas_estan_probadas():
    publicas = {
        nombre
        for nombre, f in inspect.getmembers(graficas, inspect.isfunction)
        if f.__module__ == graficas.__name__ and not nombre.startswith("_")
    }
    assert publicas == set(LLAMADAS)


@pytest.mark.parametrize("nombre", LLAMADAS)
def test_cada_grafica_es_valida(nombre, artefactos_app):
    grafica = LLAMADAS[nombre](artefactos_app)
    assert isinstance(grafica, alt.Chart | alt.LayerChart)
    especificacion = grafica.to_dict()  # valida contra el esquema de Vega-Lite
    assert _marcas(especificacion)


@pytest.mark.parametrize(
    ("nombre", "intervalo"),
    [
        ("pronostico", "area"),
        ("efectos_calendario", "errorbar"),
        ("estimaciones", "errorbar"),
        ("efecto_por_categoria", "errorbar"),
    ],
)
def test_los_intervalos_van_como_banda_o_barra_de_error(nombre, intervalo, artefactos_app):
    assert intervalo in _marcas(LLAMADAS[nombre](artefactos_app).to_dict())


def test_las_barras_de_error_van_del_extremo_bajo_al_alto_del_ic(artefactos_app):
    robustez = _leer(artefactos_app, "causal/promociones.json")["robustez"]
    especificacion = graficas.estimaciones(robustez).to_dict()
    capa, filas = _capa(especificacion, "errorbar")
    assert (capa["encoding"]["x"]["field"], capa["encoding"]["x2"]["field"]) == ("bajo", "alto")
    # La prueba de permutaciones no tiene efecto ni IC: no aparece
    assert {(f["nombre"], f["bajo"], f["alto"]) for f in filas} == {
        (n, *e["ic95_pct"]) for n, e in robustez.items() if "efecto_pct" in e
    }


def test_la_banda_del_pronostico_va_del_p10_al_p90(artefactos_app):
    especificacion = LLAMADAS["pronostico"](artefactos_app).to_dict()
    capa, filas = _capa(especificacion, "area")
    assert (capa["encoding"]["y"]["field"], capa["encoding"]["y2"]["field"]) == ("p10", "p90")
    serie = _serie(artefactos_app).filter(pl.col("modelo") == "lightgbm")
    assert sorted((f["p10"], f["p90"]) for f in filas) == sorted(zip(serie["p10"], serie["p90"], strict=True))


def test_el_pronostico_fija_colores_distintos_para_real_campeon_y_retador(artefactos_app):
    especificacion = LLAMADAS["pronostico"](artefactos_app).to_dict()
    for capa in especificacion["layer"]:
        escala = capa["encoding"]["color"]["scale"]
        assert escala["domain"] == ["real", "campeón · sistema_actual", "retador · lightgbm"]
        assert escala["range"] == list(graficas.COLORES_PRONOSTICO.values())
    assert len(set(graficas.COLORES_PRONOSTICO.values())) == 3


@pytest.mark.parametrize("metrica", ["wape_pct", "sesgo_pct"])
def test_la_metrica_por_segmento_es_la_semanal_de_metricas_json(metrica, artefactos_app):
    metricas = _leer(artefactos_app, "backtest/metricas.json")
    # Por omisión, la granularidad semanal: la métrica principal
    grafica = graficas.metrica_por_segmento(metricas, "evento", metrica)
    _, filas = _capa(grafica.to_dict(), "bar")
    esperadas = _esperadas(metricas, ("evento",), metrica, "semanal")
    assert esperadas
    assert {(f["nivel"], f["modelo"], f["valor"]) for f in filas} == esperadas


def test_la_cobertura_es_la_diaria_de_metricas_json_con_el_80_nominal(artefactos_app):
    metricas = _leer(artefactos_app, "backtest/metricas.json")
    especificacion = graficas.cobertura_intervalos(metricas).to_dict()
    _, barras = _capa(especificacion, "bar")
    esperadas = _esperadas(metricas, ("global", "evento"), "cobertura_80_pct", "diaria")
    assert esperadas
    assert {(f["nivel"], f["modelo"], f["valor"]) for f in barras} == esperadas
    # La regla es la cobertura nominal, no un intervalo
    _, nominal = _capa(especificacion, "rule")
    assert nominal == [{"x": 80.0}]


def test_la_fuga_marca_el_dia_y_los_siguientes_de_la_ventana(artefactos_app):
    reporte = _leer(artefactos_app, "calidad/reporte.json")  # ventana de t−5 a t+1; pesos de −1 a +1
    _, filas = graficas.pesos_fuga(reporte).to_dict()["datasets"].popitem()
    zonas = {f["etiqueta"]: f["zona"] for f in filas}
    assert zonas == {
        "t-1": "ventana: días previos",
        "t+0": "ventana: el día y después (fuga)",
        "t+1": "ventana: el día y después (fuga)",
    }


def test_el_pronostico_trae_el_real_y_la_banda_solo_del_modelo_con_intervalo(artefactos_app):
    especificacion = LLAMADAS["pronostico"](artefactos_app).to_dict()
    datos = especificacion["datasets"]
    banda, lineas = (datos[capa["data"]["name"]] for capa in especificacion["layer"])
    assert {f["serie"] for f in banda} == {"retador · lightgbm"}  # sistema_actual es puntual
    assert {f["serie"] for f in lineas} == {"real", "campeón · sistema_actual", "retador · lightgbm"}
    reales = [f for f in lineas if f["serie"] == "real"]
    assert len(reales) == len({f["date"] for f in reales}) == 14  # un real por día, en los 2 cortes


def test_el_equilibrio_sale_del_descuento_elegido(artefactos_app):
    promociones = _leer(artefactos_app, "causal/promociones.json")
    especificacion = graficas.efecto_por_categoria(promociones, 0.1).to_dict()
    lineas = [
        f
        for nombre, filas in especificacion["datasets"].items()
        for f in filas
        if f.get("marca") == "lift de equilibrio"
    ]
    esperado = promociones["negocio"]["por_categoria"]["Bebidas"]["por_descuento"][1]["lift_equilibrio_pct"]
    assert [(f["nombre"], f["lift"]) for f in lineas] == [("Bebidas", esperado)]


def _alertas(monitor: dict, capa: str) -> list[dict]:
    return _capa(graficas.alertas_por_semana(monitor, capa).to_dict(), "bar")[1]


def test_las_alertas_separan_capa_y_clasificacion(artefactos_app):
    monitor = _leer(artefactos_app, "drift/monitor.json")
    # Una fila por semana y clasificación. La alerta de calidad es inesperada, en el segundo corte; la de
    # entradas es esperada, en el primero
    calidad = _alertas(monitor, "calidad")
    assert len(calidad) == 2 * len(monitor["semanas"])
    assert [(f["origen"], f["clasificacion"]) for f in calidad if f["alertas"]] == [
        ("2023-11-09", "inesperada")
    ]
    entradas = [(f["origen"], f["clasificacion"]) for f in _alertas(monitor, "entradas") if f["alertas"]]
    assert entradas == [("2023-11-02", "esperada")]
    assert not [f for f in _alertas(monitor, "desempeno") if f["alertas"]]


def test_las_alertas_por_capa_suman_el_resumen_por_semana_del_monitor(artefactos_app):
    monitor = _leer(artefactos_app, "drift/monitor.json")
    por_semana = {
        (s["origen"], c): s[f"{c}s"]
        for s in monitor["resumen"]["por_semana"]
        for c in graficas.CLASIFICACIONES
    }
    suma = dict.fromkeys(por_semana, 0)
    for capa in graficas.CAPAS:
        for f in _alertas(monitor, capa):
            suma[f["origen"], f["clasificacion"]] += f["alertas"]
    assert suma == por_semana


def test_la_marca_del_backtest_va_en_su_primer_corte(artefactos_app):
    monitor = _leer(artefactos_app, "drift/monitor.json")
    _, marca = _capa(graficas.alertas_por_semana(monitor, "calidad").to_dict(), "rule")
    primero = next(s["origen"] for s in monitor["semanas"] if s["periodo"] == "backtest")
    assert [f["origen"] for f in marca] == [primero] == ["2023-11-02"]
