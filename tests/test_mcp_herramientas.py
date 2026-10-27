"""Las herramientas de solo lectura sobre artefactos sintéticos: claves, dominios de los argumentos y el
mensaje cuando falta un artefacto."""

import json

import pytest

from demanda.mcp_server import herramientas
from demanda.mcp_server.herramientas import ArgumentoInvalido, ArtefactoFaltante

CLAVES = {
    "get_data_quality": {"contratos", "malla", "conteos", "nulos", "fuga_replenishment_signal"},
    "get_eda_summary": {"demanda", "promociones", "tiendas", "sistema_actual", "implicaciones"},
    "get_metrics": {"segmento", "niveles", "cobertura_evaluacion", "seleccion"},
    "get_forecast": {"store_id", "category", "origen", "campeon", "retador", "dias"},
    "get_drift_alerts": {"resumen", "semanas_con_inesperadas", "vigilancia_retador", "validacion"},
    "get_decision_impact": {
        "politicas",
        "impacto_global",
        "impacto_por_evento",
        "pruebas_costo",
        "sensibilidad",
        "conclusion",
    },
    "get_promo_effect": {
        "balance",
        "efecto_principal",
        "canales",
        "heterogeneidad",
        "dinamica",
        "potencia",
        "negocio",
        "conclusion",
    },
}
ARGUMENTOS = {"get_forecast": {"store_id": "STR_001", "category": "Bebidas"}}


@pytest.mark.parametrize("nombre", list(herramientas.HERRAMIENTAS))
def test_cada_herramienta_devuelve_sus_claves(artefactos_e7, nombre):
    r = herramientas.llamar(artefactos_e7, nombre, ARGUMENTOS.get(nombre))
    assert CLAVES[nombre] <= set(r)


def test_metricas_por_segmento_y_regla_de_campeon(artefactos_e7):
    r = herramientas.get_metrics(artefactos_e7, "evento")
    assert set(r["niveles"]) == {"resto", "buen_fin"}
    resto = r["niveles"]["resto"]
    assert resto["modelos"]["lightgbm"]["semanal_wape_pct"] == 11.3004
    assert resto["pares_semanales"] == {"evaluados": 90, "total": 112}
    assert (r["seleccion"]["campeon"], r["seleccion"]["retador"]) == ("sistema_actual", "lightgbm")


def test_pronostico_del_ultimo_corte_con_campeon_y_retador(artefactos_e7):
    r = herramientas.get_forecast(artefactos_e7, "STR_002", "Abarrotes")
    assert r["origen"] == "2023-11-09"
    assert [d["h"] for d in r["dias"]] == list(range(1, 8))
    dia = r["dias"][0]
    assert dia["campeon"]["p10"] is None  # el sistema actual no tiene intervalo
    assert dia["retador"]["p10"] == pytest.approx(0.8 * dia["retador"]["yhat"], abs=0.01)
    assert (
        herramientas.get_forecast(artefactos_e7, "STR_002", "Abarrotes", "2023-11-02")["origen"]
        == "2023-11-02"
    )


def test_drift_resumen_y_detalle_de_semana(artefactos_e7):
    r = herramientas.get_drift_alerts(artefactos_e7)
    assert [s["origen"] for s in r["semanas_con_inesperadas"]] == ["2023-11-09"]
    assert r["semanas_con_inesperadas"][0]["recomendaciones"][0]["por"] == ["tienda=STR_002"]
    semana = herramientas.get_drift_alerts(artefactos_e7, semana="2023-11-09")
    assert semana["alertas"][0]["chequeo"] == "tienda_sin_datos"


@pytest.mark.parametrize(
    ("nombre", "argumentos", "campo"),
    [
        ("get_metrics", {"segmento": "pais"}, "segmento"),
        ("get_forecast", {"store_id": "STR_999", "category": "Bebidas"}, "store_id"),
        ("get_forecast", {"store_id": "STR_001", "category": "Juguetes"}, "category"),
        ("get_forecast", {"store_id": "STR_001", "category": "Bebidas", "origen": "2030-01-01"}, "origen"),
        ("get_drift_alerts", {"semana": "2030-01-01"}, "semana"),
    ],
)
def test_argumento_fuera_del_catalogo_da_error(artefactos_e7, nombre, argumentos, campo):
    with pytest.raises(ArgumentoInvalido, match=f"{campo} inválido"):
        herramientas.llamar(artefactos_e7, nombre, argumentos)


@pytest.mark.parametrize(
    ("nombre", "archivo", "comando"),
    [
        ("get_data_quality", "calidad/reporte.json", "make calidad"),
        ("get_eda_summary", "eda/resumen.json", "make eda"),
        ("get_metrics", "modelos/seleccion.json", "make seleccion"),
        ("get_forecast", "backtest/pronosticos.parquet", "make backtest"),
        ("get_drift_alerts", "drift/monitor.json", "make drift"),
        ("get_decision_impact", "decision/impacto.json", "make decision"),
        ("get_promo_effect", "causal/promociones.json", "make causal"),
    ],
)
def test_artefacto_faltante_dice_que_comando_lo_genera(artefactos_e7, nombre, archivo, comando):
    (artefactos_e7 / archivo).unlink()
    with pytest.raises(ArtefactoFaltante, match=comando) as e:
        herramientas.llamar(artefactos_e7, nombre, ARGUMENTOS.get(nombre))
    assert str(artefactos_e7) not in str(e.value)  # el mensaje no expone rutas


def test_herramienta_desconocida(artefactos_e7):
    with pytest.raises(ArgumentoInvalido, match="herramienta"):
        herramientas.llamar(artefactos_e7, "borrar_todo")


def test_una_recomendacion_con_muchas_alertas_se_cuenta_sin_identificadores(artefactos_e7):
    semana = herramientas.get_drift_alerts(artefactos_e7)["semanas_con_inesperadas"][0]
    reentrenar = next(r for r in semana["recomendaciones"] if r["codigo"] == "reentrenar")
    assert set(reentrenar) == {"codigo", "texto", "alertas", "por_chequeo"}
    assert reentrenar["alertas"] == 5
    assert reentrenar["por_chequeo"] == {"nivel_demanda": 2, "tienda": 3}
    assert "STR_" not in json.dumps(reentrenar)
