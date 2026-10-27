"""Los agentes con un `FunctionModel`: citar una cifra real pasa, inventar reintenta y, si insiste, cae a
la plantilla con el motivo; un rechazo del modelo también cae. Y el proveedor: modo de salida, ajustes sin
parámetros de muestreo, precios y keys."""

import pytest
from pydantic import SecretStr
from pydantic_ai import NativeOutput
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from demanda.agentes import etapa, prompts, proveedor
from demanda.agentes.modelos import DOMINIOS, ORDEN, SalidaDominio
from demanda.config import Config


def _correr(config_e7, tmp_path, modelo):
    return etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json", modelo=modelo.modelo())


def test_agente_que_cita_una_cifra_real_pasa(config_e7, tmp_path, simulado):
    r = _correr(config_e7, tmp_path, simulado.Modelo())
    assert r["modo"] == "vivo"
    for dominio in ORDEN:
        d = r["dominios"][dominio]
        assert (d["origen"], d["reintentos"], d["motivo"]) == ("vivo", 0, None)
        assert d["validacion"]["valida"]
    # cada dominio llamó a sus herramientas y solo a ellas
    for dominio, nombres in DOMINIOS.items():
        assert {ll["herramienta"] for ll in r["dominios"][dominio]["llamadas"]} == set(nombres)


def test_agente_que_inventa_reintenta_y_corrige(config_e7, tmp_path, simulado):
    r = _correr(config_e7, tmp_path, simulado.Modelo(citar=simulado.inventa_y_corrige))
    d = r["dominios"]["datos"]
    assert (d["origen"], d["reintentos"]) == ("vivo", 1)
    assert {x["cifra"] for x in d["rechazadas_previas"]} == {simulado.inventada}


def test_agente_que_insiste_cae_a_la_plantilla_con_el_motivo(config_e7, tmp_path, simulado):
    r = _correr(config_e7, tmp_path, simulado.Modelo(citar=simulado.inventa))
    for dominio in ORDEN:
        d = r["dominios"][dominio]
        assert d["origen"] == "plantilla"
        assert d["motivo"] == "el validador rechazó cifras también en el reintento"
        assert d["reintentos"] == 1
        assert {x["cifra"] for x in d["rechazadas_previas"]} == {simulado.inventada}
        assert d["validacion"]["valida"]  # la plantilla sí pasa
    assert not (tmp_path / "cache.json").exists()  # nada en vivo: la caché no se toca


def test_rechazo_del_modelo_cae_a_la_plantilla(config_e7, tmp_path, simulado):
    r = _correr(config_e7, tmp_path, simulado.Modelo(rechazo=True))
    for dominio in ORDEN:
        d = r["dominios"][dominio]
        assert d["origen"] == "plantilla"
        assert "rechazó la solicitud (cyber)" in d["motivo"]


def test_un_modelo_real_sin_red_cae_a_la_plantilla(config_e7, tmp_path):
    # ALLOW_MODEL_REQUESTS = False: un AnthropicModel de verdad falla antes de salir a la red
    modelo = AnthropicModel("claude-opus-5-5", provider=AnthropicProvider(api_key="sk-prueba"))
    r = etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json", modelo=modelo)
    assert all(r["dominios"][d]["origen"] == "plantilla" for d in ORDEN)
    assert "ALLOW_MODEL_REQUESTS" in r["dominios"]["datos"]["motivo"]
    assert r["gasto"]["corrida"]["solicitudes"] == 0


# --- proveedor -----------------------------------------------------------------------------------------


def test_opus_5_5_usa_salida_nativa():
    opus = AnthropicModel("claude-opus-5-5", provider=AnthropicProvider(api_key="sk-prueba"))
    assert proveedor.modo_salida(opus) == "native"
    assert isinstance(proveedor.tipo_salida(opus, SalidaDominio), NativeOutput)
    # Claude Opus 5 sí acepta el tool_choice forzado: se deja el modo automático de Pydantic AI
    opus_5 = AnthropicModel("claude-opus-5", provider=AnthropicProvider(api_key="sk-prueba"))
    assert proveedor.tipo_salida(opus_5, SalidaDominio) is SalidaDominio


def test_ajustes_sin_muestreo_con_esfuerzo_y_respaldo():
    a = proveedor.ajustes("anthropic:claude-opus-5-5")
    assert not {"temperature", "top_p", "top_k"} & set(a)
    assert a.get("anthropic_effort") == "medium"
    assert a.get("anthropic_betas") == ["server-side-fallback-2026-07-01"]
    assert a.get("extra_body") == {"fallbacks": "default"}
    # a otros proveedores no les llega el cuerpo extra de Anthropic
    otro = proveedor.ajustes("openai:gpt-6-sol")
    assert set(otro) == {"max_tokens"}


@pytest.mark.parametrize(
    ("modelo", "esperado"),
    [
        ("anthropic:claude-opus-5-5", (4.0, 20.0)),
        ("openrouter:anthropic/claude-opus-5.5", (4.0, 20.0)),
        ("claude-sonnet-5-5", (2.0, 10.0)),
        ("openai:modelo-desconocido", (10.0, 50.0)),  # el precio más alto de la tabla
    ],
)
def test_precio_por_modelo(modelo, esperado):
    assert proveedor.precio(modelo) == esperado


def test_keys_vacias_no_cuentan_y_no_se_crea_el_modelo():
    sin = Config(_env_file=None, anthropic_api_key=SecretStr(" "))
    assert not proveedor.hay_key(sin)
    with pytest.raises(ValueError, match="falta la key de anthropic"):
        proveedor.crear_modelo(sin)
    con = Config(_env_file=None, anthropic_api_key=SecretStr("sk-prueba"))
    assert proveedor.hay_key(con)
    assert isinstance(proveedor.crear_modelo(con), AnthropicModel)
    otro = Config(
        _env_file=None, llm_model="openrouter:anthropic/claude-opus-5.5", anthropic_api_key=SecretStr("x")
    )
    assert not proveedor.hay_key(otro)  # cada proveedor usa su propia key
    with pytest.raises(ValueError, match="proveedor no soportado"):
        proveedor.crear_modelo(Config(_env_file=None, llm_model="mistral:x"))


def test_precio_maximo_es_el_mas_alto_de_la_tabla():
    assert proveedor.PRECIO_MAXIMO == (10.0, 50.0)


# --- límites, historial y salidas inválidas ------------------------------------------------------------


def _es_reintento(messages) -> bool:
    return any(
        isinstance(p, UserPromptPart) and "El validador rechazó" in str(p.content)
        for m in messages
        for p in m.parts
    )


def _desde_el_ultimo_prompt(messages) -> list:
    i = max(n for n, m in enumerate(messages) if any(isinstance(p, UserPromptPart) for p in m.parts))
    return messages[i:]


def _salida(info, cifra: str, herramienta: str) -> ModelResponse:
    insight = {
        "titulo": "t",
        "hallazgo": f"La cifra es {cifra}.",
        "evidencia": [{"cifra": cifra, "que_mide": "x", "herramienta": herramienta}],
        "accion": "a",
        "confianza": "media",
    }
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"insights": [insight, insight]})])


def test_los_numeros_de_un_error_de_herramienta_no_cuentan(config_e7, tmp_path, simulado):
    # El agente pide un segmento inventado: el error de la herramienta repite «777.7», que no es una cifra
    def fn(messages, info):
        nombres = [t.name for t in info.function_tools]
        if "get_metrics" not in nombres:
            return simulado.Modelo()(messages, info)
        if not any(isinstance(p, ToolReturnPart) for m in _desde_el_ultimo_prompt(messages) for p in m.parts):
            return ModelResponse(parts=[ToolCallPart("get_metrics", {"segmento": "777.7"}, tool_call_id="m")])
        return _salida(info, "777.7", "get_metrics")

    r = etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json", modelo=FunctionModel(fn))
    d = r["dominios"]["pronostico"]
    assert d["origen"] == "plantilla"
    assert {x["cifra"] for x in d["rechazadas_previas"]} == {"777.7"}


def test_el_limite_de_solicitudes_cuenta_el_intento_y_el_reintento(config_e7, tmp_path, simulado):
    # datos gasta 5 solicitudes en el intento (4 llamadas y una salida inventada) y 4 en el reintento:
    # juntas pasan el límite de 8
    def fn(messages, info):
        nombres = [t.name for t in info.function_tools]
        if "get_data_quality" not in nombres:
            return simulado.Modelo()(messages, info)
        reintento = _es_reintento(messages)
        hechas = sum(
            isinstance(p, ToolReturnPart) and p.tool_name in nombres
            for m in _desde_el_ultimo_prompt(messages)
            for p in m.parts
        )
        if hechas < (3 if reintento else 4):
            return ModelResponse(
                parts=[ToolCallPart(nombres[hechas % 2], {}, tool_call_id=f"t{len(messages)}")]
            )
        cifra = simulado.cita_real(messages, info) if reintento else simulado.inventada
        return _salida(info, cifra, "get_data_quality")

    r = etapa.ejecutar(config_e7, "vivo", cache_path=tmp_path / "cache.json", modelo=FunctionModel(fn))
    d = r["dominios"]["datos"]
    assert d["origen"] == "plantilla"
    assert d["motivo"].startswith("límite de uso del agente")
    assert d["gasto"]["solicitudes"] == 8


def test_el_reintento_recibe_el_historial(config_e7, tmp_path, simulado):
    modelo = simulado.Modelo(citar=simulado.inventa_y_corrige)
    _correr(config_e7, tmp_path, modelo)
    primera_del_reintento = next(m for m in modelo.vistos if _es_reintento(m[-1:]))
    previas = primera_del_reintento[:-1]
    assert any(isinstance(p, ToolReturnPart) for m in previas for p in m.parts)
    assert any(
        isinstance(p, UserPromptPart) and p.content == prompts.USUARIO for m in previas for p in m.parts
    )


def test_un_agente_que_entrega_un_solo_insight_cae_a_la_plantilla(config_e7, tmp_path, simulado):
    r = _correr(config_e7, tmp_path, simulado.Modelo(insights=1))
    for dominio in ORDEN:
        d = r["dominios"][dominio]
        assert d["origen"] == "plantilla"
        assert d["motivo"].startswith("error del modelo")
