"""La etapa de insights: modos, caché y presupuesto, siempre sin red ni key."""

import json

import pytest

from demanda.agentes import cache, etapa, orquestador, proveedor
from demanda.agentes.modelos import (
    DOMINIOS,
    LLAMADAS_POR_OMISION,
    ORDEN,
    Evidencia,
    Insight,
    ResultadoDominio,
)
from demanda.agentes.presupuesto import LIMITES
from demanda.cli import main
from demanda.mcp_server import herramientas


def _vivo(config_e7, cache_path, modelo):
    return etapa.ejecutar(config_e7, "vivo", cache_path=cache_path, modelo=modelo.modelo())


def _sin_red(monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("no debe crear un modelo")

    monkeypatch.setattr(proveedor, "crear_modelo", prohibido)


# --- plantillas y modo auto sin key --------------------------------------------------------------------


def test_plantillas_pasan_el_validador(artefactos_e7):
    r = orquestador.plantillas_todas(artefactos_e7)
    assert list(r) == list(ORDEN)
    for resultado in r.values():
        assert 2 <= len(resultado.insights) <= 4
        assert resultado.validacion.valida and resultado.validacion.cifras > 0


def test_la_plantilla_del_monitor_cita_el_conteo_de_la_herramienta(artefactos_e7):
    r = orquestador.plantillas_todas(artefactos_e7)
    monitor = next(i for i in r["pronostico"].insights if i.titulo.startswith("El monitor"))
    resumen = herramientas.get_drift_alerts(artefactos_e7)["resumen"]
    assert monitor.evidencia[0].cifra == str(resumen["backtest"]["por_capa"]["entradas"]["inesperadas"])


def test_auto_sin_key_usa_plantillas_y_es_determinista(config_e7, tmp_path, monkeypatch):
    _sin_red(monkeypatch)
    r = etapa.ejecutar(config_e7, "auto", cache_path=tmp_path / "no_existe.json")
    assert (r["modo"], r["aviso"]) == ("cache", "sin key para anthropic: se usa la caché")
    assert list(r["dominios"]) == list(ORDEN)
    assert all(d["origen"] == "plantilla" for d in r["dominios"].values())
    assert r["validacion"]["todas_validas"] and r["validacion"]["rechazadas"] == 0
    salida = config_e7.artifacts_dir / "insights" / "insights.json"
    primero = salida.read_bytes()
    etapa.ejecutar(config_e7, "auto", cache_path=tmp_path / "no_existe.json")
    assert salida.read_bytes() == primero
    assert not (config_e7.artifacts_dir / "insights" / "gasto.json").exists()


def test_vivo_sin_key_es_un_error(config_e7, tmp_path):
    with pytest.raises(ValueError, match="requiere la key de anthropic"):
        etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json")


def test_falta_un_artefacto(config_e7, tmp_path):
    (config_e7.artifacts_dir / "drift" / "monitor.json").unlink()
    with pytest.raises(FileNotFoundError, match="make drift"):
        etapa.ejecutar(config_e7, "plantilla", cache_path=tmp_path / "cache.json")


# --- caché ---------------------------------------------------------------------------------------------


def test_vivo_escribe_la_cache_y_auto_sin_key_la_usa(config_e7, tmp_path, monkeypatch, simulado):
    cache_path = tmp_path / "cache.json"
    _vivo(config_e7, cache_path, simulado.Modelo())
    contenido = json.loads(cache_path.read_text())
    assert list(contenido["dominios"]) == list(ORDEN)
    entrada = contenido["dominios"]["datos"]
    assert {
        "insights",
        "modelo",
        "prompt_version",
        "prompts_sha256",
        "huella",
        "llamadas",
        "artefactos_sha256",
    } <= set(entrada)
    assert set(entrada["artefactos_sha256"]) == {"calidad/reporte.json", "eda/resumen.json"}

    _sin_red(monkeypatch)
    r = etapa.ejecutar(config_e7, "auto", cache_path=cache_path)
    assert r["modo"] == "cache"
    assert all(r["dominios"][d]["origen"] == "cache" for d in ORDEN)
    assert r["validacion"]["todas_validas"]


def test_cache_invalida_por_version_del_prompt(config_e7, tmp_path, simulado):
    cache_path = tmp_path / "cache.json"
    _vivo(config_e7, cache_path, simulado.Modelo())
    contenido = json.loads(cache_path.read_text())
    contenido["dominios"]["causal"]["prompt_version"] = "e7.0"
    cache_path.write_text(json.dumps(contenido))
    r = etapa.ejecutar(config_e7, "cache", cache_path=cache_path)
    assert r["dominios"]["datos"]["origen"] == "cache"
    assert r["dominios"]["causal"]["origen"] == "plantilla"
    assert "versión de prompts e7.0" in r["dominios"]["causal"]["motivo"]
    assert r["dominios"]["orquestador"]["motivo"] == "algún dominio no salió de la caché"


def _entrada_datos(artefactos, cifra: str) -> dict:
    """Una entrada de caché de `datos` que cita `cifra` de get_data_quality."""
    respuestas = cache.responder(artefactos, LLAMADAS_POR_OMISION["datos"])
    insight = Insight(
        titulo="Fuga",
        hallazgo=f"La señal correlaciona {cifra} con la media móvil.",
        evidencia=[Evidencia(cifra=cifra, que_mide="correlación", herramienta="get_data_quality")],
        accion="Rezagar la señal.",
        confianza="alta",
    )
    resultado = ResultadoDominio(
        origen="vivo", insights=[insight, insight], validacion={"valida": True, "cifras": 2}
    )
    return cache.entrada(artefactos, resultado, respuestas)


def _cambiar_corr(artefactos, corr: float) -> None:
    ruta = artefactos / "calidad" / "reporte.json"
    reporte = json.loads(ruta.read_text())
    reporte["fuga_replenishment_signal"]["mejor_ventana"]["corr"] = corr
    ruta.write_text(json.dumps(reporte))


def test_cache_vale_con_los_mismos_artefactos_y_con_cambios_de_1e_7(artefactos_e7):
    entrada = _entrada_datos(artefactos_e7, "0.9312")
    assert cache.vigente(artefactos_e7, "datos", entrada)[0] is not None
    _cambiar_corr(artefactos_e7, 0.9312 + 1e-7)  # ruido de plataforma: ni la huella ni la cifra cambian
    resultado, motivo, _ = cache.vigente(artefactos_e7, "datos", entrada)
    assert resultado is not None and resultado.origen == "cache", motivo


def test_cache_deja_de_valer_si_cambia_una_cifra_que_cita(artefactos_e7):
    entrada = _entrada_datos(artefactos_e7, "0.9312")
    _cambiar_corr(artefactos_e7, 0.9314)  # a 3 cifras sigue siendo 0.931: la huella no lo ve
    resultado, motivo, _ = cache.vigente(artefactos_e7, "datos", entrada)
    assert resultado is None
    assert motivo == "el validador rechaza cifras de la caché contra las herramientas actuales"


def test_cache_deja_de_valer_si_cambia_la_huella(artefactos_e7):
    entrada = _entrada_datos(artefactos_e7, "0.9312")
    ruta = artefactos_e7 / "calidad" / "reporte.json"
    reporte = json.loads(ruta.read_text())
    reporte["conteos"]["pct_inconsistentes"] = 31.5
    ruta.write_text(json.dumps(reporte))
    assert cache.vigente(artefactos_e7, "datos", entrada)[1] == "la huella de los datos cambió"


def test_la_huella_redondea_a_3_cifras():
    assert cache.redondear({"a": 0.931234, "b": [123456.7, 2], "c": True}) == {
        "a": 0.931,
        "b": [123000.0, 2],
        "c": True,
    }


# --- presupuesto ---------------------------------------------------------------------------------------


def test_limites_por_agente():
    assert (LIMITES.request_limit, LIMITES.total_tokens_limit) == (8, 150_000)


def test_presupuesto_agotado_niega_el_modo_vivo(config_e7, tmp_path, simulado):
    gasto = config_e7.artifacts_dir / "insights" / "gasto.json"
    gasto.parent.mkdir(parents=True)
    gasto.write_text(json.dumps({"acumulado_usd": 20.0, "corridas": []}))
    modelo = simulado.Modelo()
    r = etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json", modelo=modelo.modelo())
    assert r["modo"] == "cache"
    assert "LLM_PRESUPUESTO_USD" in r["aviso"]
    assert modelo.solicitudes == 0
    assert all(d["origen"] == "plantilla" for d in r["dominios"].values())


def test_la_corrida_se_corta_al_pasar_el_tope(config_e7, tmp_path, simulado):
    # 10 mil de entrada y 60 mil de salida: USD 1.24 con Opus 5.5, más que el tope de USD 1 en una respuesta
    modelo = simulado.Modelo(tokens=(10_000, 60_000))
    r = _vivo(config_e7, tmp_path / "cache.json", modelo)
    assert modelo.solicitudes == 1
    for dominio in ORDEN:
        d = r["dominios"][dominio]
        assert d["origen"] == "plantilla"
        assert "tope de USD 1" in d["motivo"]
    assert r["gasto"]["corrida"]["costo_usd"] == pytest.approx(1.24)


def test_el_gasto_se_acumula_entre_corridas(config_e7, tmp_path, simulado):
    cache_path = tmp_path / "cache.json"
    primera = _vivo(config_e7, cache_path, simulado.Modelo())
    segunda = _vivo(config_e7, cache_path, simulado.Modelo())
    costo = primera["gasto"]["corrida"]["costo_usd"]
    # 3 dominios × 2 solicitudes y 1 del orquestador, a USD 0.014 cada una
    assert costo == pytest.approx(7 * (1_000 * 4 + 500 * 20) / 1e6)
    assert segunda["gasto"]["acumulado_usd"] == pytest.approx(2 * costo)
    gasto = json.loads((config_e7.artifacts_dir / "insights" / "gasto.json").read_text())
    assert len(gasto["corridas"]) == 2
    assert set(gasto["corridas"][0]["por_dominio"]) == set(ORDEN)


# --- CLI -----------------------------------------------------------------------------------------------


def test_cli_insights_en_modo_plantilla(config_e7, monkeypatch, capsys):
    monkeypatch.setattr("demanda.cli.get_config", lambda: config_e7)
    main(["insights", "--modo", "plantilla"])
    salida = capsys.readouterr().out
    assert "modo: plantilla" in salida
    for dominio in (*DOMINIOS, "orquestador"):
        assert dominio in salida
    assert (config_e7.artifacts_dir / "insights" / "insights.json").exists()


# --- caché: huella, corridas y dominio -----------------------------------------------------------------


def _pct_inconsistentes(artefactos, valor: float) -> None:
    ruta = artefactos / "calidad" / "reporte.json"
    reporte = json.loads(ruta.read_text())
    reporte["conteos"]["pct_inconsistentes"] = valor
    ruta.write_text(json.dumps(reporte))


def test_la_huella_ve_cambios_en_la_tercera_cifra(artefactos_e7):
    llamadas = LLAMADAS_POR_OMISION["datos"]
    _pct_inconsistentes(artefactos_e7, 19.96)
    antes = cache.huella(cache.responder(artefactos_e7, llamadas))
    _pct_inconsistentes(artefactos_e7, 21.0)
    assert cache.huella(cache.responder(artefactos_e7, llamadas)) != antes


def test_un_dominio_que_reintento_sale_de_la_cache(config_e7, tmp_path, simulado):
    # En el reintento el agente vuelve a llamar sus herramientas: la huella no debe contar las repetidas
    cache_path = tmp_path / "cache.json"
    r = _vivo(config_e7, cache_path, simulado.Modelo(citar=simulado.inventa_y_corrige))
    assert (r["dominios"]["datos"]["origen"], r["dominios"]["datos"]["reintentos"]) == ("vivo", 1)
    r2 = etapa.ejecutar(config_e7, "cache", cache_path=cache_path)
    assert all(r2["dominios"][d]["origen"] == "cache" for d in ORDEN), {
        d: v["motivo"] for d, v in r2["dominios"].items()
    }


def _datos_inventa(simulado):
    """Un modelo en el que solo `datos` inventa, en el intento y en el reintento."""

    def citar(messages, info):
        nombres = {t.name for t in info.function_tools}
        return simulado.inventada if "get_data_quality" in nombres else simulado.cita_real(messages, info)

    return simulado.Modelo(citar=citar)


def test_una_corrida_parcial_no_mezcla_entradas_de_otra_corrida(config_e7, tmp_path, simulado):
    cache_path = tmp_path / "cache.json"
    _vivo(config_e7, cache_path, simulado.Modelo())
    r = _vivo(config_e7, cache_path, _datos_inventa(simulado))
    assert r["dominios"]["datos"]["origen"] == "plantilla"
    # datos no deja la entrada de la corrida anterior, y sin los tres dominios no se guarda el orquestador
    assert list(json.loads(cache_path.read_text())["dominios"]) == ["pronostico", "causal"]
    r2 = etapa.ejecutar(config_e7, "cache", cache_path=cache_path)
    assert (r2["dominios"]["datos"]["origen"], r2["dominios"]["datos"]["motivo"]) == (
        "plantilla",
        "no hay entrada en la caché",
    )
    assert r2["dominios"]["pronostico"]["origen"] == "cache"
    assert r2["dominios"]["orquestador"]["motivo"] == "algún dominio no salió de la caché"


def test_una_corrida_sin_dominios_en_vivo_no_toca_la_cache(config_e7, tmp_path, simulado):
    cache_path = tmp_path / "cache.json"
    _vivo(config_e7, cache_path, simulado.Modelo())
    antes = cache_path.read_bytes()
    _vivo(config_e7, cache_path, simulado.Modelo(rechazo=True))
    assert cache_path.read_bytes() == antes


def test_el_orquestador_solo_vale_sobre_las_entradas_que_resumio(config_e7, tmp_path, simulado):
    cache_path = tmp_path / "cache.json"
    _vivo(config_e7, cache_path, simulado.Modelo())
    contenido = json.loads(cache_path.read_text())
    contenido["dominios"]["datos"]["insights"][0]["titulo"] = "Otro título, sin cifras"
    cache_path.write_text(json.dumps(contenido))
    r = etapa.ejecutar(config_e7, "cache", cache_path=cache_path)
    assert r["dominios"]["datos"]["origen"] == "cache"  # la entrada de datos sigue siendo válida
    assert (r["dominios"]["orquestador"]["origen"], r["dominios"]["orquestador"]["motivo"]) == (
        "plantilla",
        "el resumen es de otras entradas de dominio",
    )


def test_una_llamada_fuera_del_dominio_invalida_la_entrada(artefactos_e7):
    entrada = _entrada_datos(artefactos_e7, "0.9312")
    entrada["llamadas"].append(
        {"herramienta": "get_forecast", "argumentos": {"store_id": "STR_002", "category": "Abarrotes"}}
    )
    resultado, motivo, _ = cache.vigente(artefactos_e7, "datos", entrada)
    assert resultado is None
    assert motivo == "la entrada trae llamadas fuera del dominio: get_forecast"


def test_cache_invalida_por_hash_de_los_prompts(artefactos_e7):
    entrada = {**_entrada_datos(artefactos_e7, "0.9312"), "prompts_sha256": "0" * 64}
    resultado, motivo, _ = cache.vigente(artefactos_e7, "datos", entrada)
    assert resultado is None
    assert motivo is not None and "hash de los prompts" in motivo


def test_un_costo_igual_al_tope_corta_la_corrida(config_e7, tmp_path, simulado):
    modelo = simulado.Modelo(tokens=(0, 50_000))  # USD 1.00 justos con Opus 5.5
    r = _vivo(config_e7, tmp_path / "cache.json", modelo)
    assert modelo.solicitudes == 1
    assert r["gasto"]["corrida"]["costo_usd"] == pytest.approx(1.0)
    assert all("tope de USD 1" in r["dominios"][d]["motivo"] for d in ORDEN)
