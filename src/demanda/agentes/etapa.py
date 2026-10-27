"""La etapa de insights: de los artefactos a `artifacts/insights/insights.json`, que lee el tablero.

Modos:
- `vivo`: los agentes con el proveedor de `LLM_MODEL`; requiere su key y reescribe `insights/cache.json`
  (`INSIGHTS_DIR` cambia la carpeta) con los dominios que salieron en vivo (ver `cache.py`: la caché es de
  una sola corrida);
- `cache`: `insights/cache.json`; cada dominio cuya entrada no vale usa su plantilla;
- `plantilla`: las conclusiones generadas en Python, sin LLM;
- `auto` (por omisión): `vivo` si hay key; si no, `cache`.

El modo vivo no arranca si el gasto acumulado de `gasto.json` ya llegó a `LLM_PRESUPUESTO_USD`: lo dice y
sigue con la caché. `insights.json` lleva el modo, el origen de cada dominio y el resultado del validador;
en los modos sin red, dos corridas sobre los mismos artefactos lo escriben igual.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai.models import Model

from demanda.agentes import cache, orquestador, proveedor
from demanda.agentes.modelos import DOMINIOS, ORDEN, ORQUESTADOR, ResultadoDominio
from demanda.agentes.presupuesto import Presupuesto, leer_gasto, registrar_corrida
from demanda.agentes.prompts import PROMPT_VERSION, hash_prompts
from demanda.config import Config
from demanda.mcp_server import herramientas

MODOS = ("auto", "vivo", "cache", "plantilla")


def requeridos(artefactos: Path) -> None:
    """Falla con el comando que falta si no está algún artefacto que leen los dominios."""
    for herramienta in {h for hs in DOMINIOS.values() for h in hs}:
        for relativa in herramientas.FUENTES[herramienta]:
            if not (artefactos / relativa).exists():
                raise FileNotFoundError(
                    f"falta el artefacto {relativa}; genéralo con `{herramientas.COMANDOS[relativa]}`"
                )


def modo_efectivo(
    modo: str, cfg: Config, gasto_path: Path, hay_modelo: bool = False
) -> tuple[str, str | None]:
    """El modo que corre de verdad y, si difiere del pedido, por qué."""
    if modo not in MODOS:
        raise ValueError(f"modo inválido: {modo}; usa {', '.join(MODOS)}")
    if modo in ("cache", "plantilla"):
        return modo, None
    proveedor_llm = proveedor.partes(cfg.llm_model)[0]
    if not (hay_modelo or proveedor.hay_key(cfg)):
        if modo == "vivo":
            raise ValueError(
                f"el modo vivo requiere la key de {proveedor_llm} para LLM_MODEL={cfg.llm_model}"
            )
        return "cache", f"sin key para {proveedor_llm}: se usa la caché"
    acumulado = leer_gasto(gasto_path)["acumulado_usd"]
    if acumulado >= cfg.llm_presupuesto_usd:
        return "cache", (
            f"el gasto acumulado (USD {acumulado:.4f}) ya alcanzó LLM_PRESUPUESTO_USD (USD "
            f"{cfg.llm_presupuesto_usd:g}): el modo vivo no arranca y se usa la caché"
        )
    return "vivo", None


def _resumen_validacion(resultados: dict[str, ResultadoDominio]) -> dict[str, Any]:
    return {
        "todas_validas": all(r.validacion.valida for r in resultados.values()),
        "cifras": sum(r.validacion.cifras for r in resultados.values()),
        "rechazadas": sum(len(r.validacion.rechazadas) for r in resultados.values()),
    }


def _escribir_cache(
    artefactos: Path,
    cache_path: Path,
    resultados: dict[str, ResultadoDominio],
    respuestas: dict[str, Any],
) -> None:
    """La caché es de una sola corrida: guarda los dominios que salieron en vivo en esta, y el orquestador
    solo si salieron los tres. Si ningún dominio salió en vivo, la caché anterior se queda como estaba."""
    nuevas = {
        d: cache.entrada(artefactos, resultados[d], respuestas[d])
        for d in DOMINIOS
        if resultados[d].origen == "vivo"
    }
    if not nuevas:
        return
    if len(nuevas) == len(DOMINIOS) and resultados[ORQUESTADOR].origen == "vivo":
        nuevas[ORQUESTADOR] = cache.entrada(
            artefactos, resultados[ORQUESTADOR], respuestas[ORQUESTADOR], entradas_de_dominio=nuevas
        )
    cache.escribir(cache_path, nuevas)


def ejecutar(
    cfg: Config,
    modo: str = "auto",
    cache_path: Path | None = None,
    modelo: Model | None = None,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Escribe `insights.json` en el modo pedido. `modelo` reemplaza al del proveedor (pruebas)."""
    artefactos = cfg.artifacts_dir
    # La caché se versiona: vive fuera de artifacts/
    cache_path = cache_path or cfg.insights_dir / "cache.json"
    salida = artefactos / "insights" / "insights.json"
    gasto_path = artefactos / "insights" / "gasto.json"
    requeridos(artefactos)
    efectivo, aviso = modo_efectivo(modo, cfg, gasto_path, hay_modelo=modelo is not None)
    gasto: dict[str, Any] | None = None

    if efectivo == "vivo":
        presupuesto = Presupuesto(tope_usd=cfg.llm_tope_corrida_usd, precio=proveedor.precio(cfg.llm_model))
        corrida = orquestador.Corrida(
            modelo=modelo or proveedor.crear_modelo(cfg),
            nombre=cfg.llm_model,
            ajustes=proveedor.ajustes(cfg.llm_model),
            presupuesto=presupuesto,
        )
        try:
            resultados, respuestas = asyncio.run(orquestador.en_vivo(artefactos, corrida, progreso))
        finally:
            # El gasto se registra aunque la corrida falle a medias
            registro = registrar_corrida(gasto_path, cfg.llm_model, presupuesto)
        _escribir_cache(artefactos, cache_path, resultados, respuestas)
        gasto = {
            "corrida": presupuesto.total().model_dump(),
            "acumulado_usd": registro["acumulado_usd"],
            "presupuesto_usd": cfg.llm_presupuesto_usd,
            "tope_corrida_usd": cfg.llm_tope_corrida_usd,
            "precio_usd_por_millon": {"entrada": presupuesto.precio[0], "salida": presupuesto.precio[1]},
        }
    elif efectivo == "cache":
        resultados = orquestador.desde_cache(artefactos, cache_path)
    else:
        resultados = orquestador.plantillas_todas(artefactos)

    insights = {
        "modo_solicitado": modo,
        "modo": efectivo,
        "aviso": aviso,
        "modelo": cfg.llm_model,
        "prompt_version": PROMPT_VERSION,
        "prompts_sha256": hash_prompts(),
        "validacion": _resumen_validacion(resultados),
        "dominios": {d: resultados[d].model_dump(mode="json") for d in ORDEN},
        "gasto": gasto,
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(insights, indent=2, ensure_ascii=False) + "\n")
    return insights
