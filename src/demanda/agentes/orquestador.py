"""Corre los dominios, valida, reintenta y cae a la plantilla; después, el resumen del orquestador.

El orquestador es código más un agente:
- los tres agentes de dominio corren en orden fijo (datos, pronóstico, promociones), cada uno con sus
  herramientas del servidor MCP en proceso;
- si el validador rechaza alguna cifra, el agente recibe la lista y reintenta una vez sobre el mismo
  historial; si vuelve a fallar, el dominio usa su plantilla y la salida guarda el motivo y las cifras;
- un rechazo del modelo, un error del proveedor, el límite de uso del agente o el tope de la corrida
  también llevan a la plantilla;
- el agente orquestador no tiene herramientas: recibe los insights ya validados y sus cifras se validan
  contra las respuestas de las herramientas de los tres dominios.

Los modos `cache` y `plantilla` no tocan la red.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import ContentFilterError, UsageLimitExceeded
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from demanda.agentes import cache, plantillas, prompts, proveedor, validador
from demanda.agentes.modelos import (
    DOMINIOS,
    LLAMADAS_POR_OMISION,
    ORQUESTADOR,
    Llamada,
    Rechazo,
    ResultadoDominio,
    SalidaDominio,
)
from demanda.agentes.presupuesto import LIMITES, Freno, Presupuesto, RechazoDelModelo, TopeDeCorrida
from demanda.mcp_server import servidor

Respuestas = list[tuple[Llamada, Any]]
MAX_ERROR = 300  # caracteres del error del proveedor que se guardan en la salida


@dataclass
class Corrida:
    """Lo que comparten los agentes de una corrida en vivo."""

    modelo: Model
    nombre: str  # LLM_MODEL, para la salida y la caché
    ajustes: ModelSettings
    presupuesto: Presupuesto


def _validar(insights: Sequence[Any], respuestas: Respuestas, dominio: str):
    permitidas = validador.Permitidas(r for _, r in respuestas)
    return validador.validar(insights, permitidas, cache.herramientas_del(dominio))


# --- plantillas ----------------------------------------------------------------------------------------


def plantilla_dominio(
    artefactos: Path, dominio: str, motivo: str | None = None
) -> tuple[ResultadoDominio, Respuestas]:
    """El dominio con su plantilla, sobre las llamadas por omisión de sus herramientas."""
    respuestas = cache.responder(artefactos, LLAMADAS_POR_OMISION[dominio])
    insights = plantillas.PLANTILLAS[dominio]({ll.herramienta: r for ll, r in respuestas})
    resultado = ResultadoDominio(
        origen="plantilla",
        insights=insights,
        validacion=_validar(insights, respuestas, dominio),
        motivo=motivo,
        llamadas=[ll for ll, _ in respuestas],
    )
    return resultado, respuestas


def plantilla_orquestador(
    por_dominio: dict[str, ResultadoDominio], respuestas: Respuestas, motivo: str | None = None
) -> ResultadoDominio:
    insights = plantillas.orquestador({d: r.insights for d, r in por_dominio.items()})
    return ResultadoDominio(
        origen="plantilla",
        insights=insights,
        validacion=_validar(insights, respuestas, ORQUESTADOR),
        motivo=motivo,
        llamadas=cache.unicas(ll for ll, _ in respuestas),
    )


def plantillas_todas(artefactos: Path, motivo: str | None = None) -> dict[str, ResultadoDominio]:
    resultados: dict[str, ResultadoDominio] = {}
    union: Respuestas = []
    for dominio in DOMINIOS:
        resultados[dominio], respuestas = plantilla_dominio(artefactos, dominio, motivo)
        union += respuestas
    resultados[ORQUESTADOR] = plantilla_orquestador(resultados, union, motivo)
    return resultados


# --- caché ---------------------------------------------------------------------------------------------


def desde_cache(artefactos: Path, cache_path: Path) -> dict[str, ResultadoDominio]:
    """Cada dominio de la caché si su entrada vale; si no, su plantilla con el motivo."""
    entradas = cache.leer(cache_path)
    resultados: dict[str, ResultadoDominio] = {}
    union: Respuestas = []
    for dominio in DOMINIOS:
        r, motivo, respuestas = cache.vigente(artefactos, dominio, entradas.get(dominio))
        if r is None:
            r, respuestas = plantilla_dominio(artefactos, dominio, motivo)
        resultados[dominio] = r
        union += respuestas
    if all(resultados[d].origen == "cache" for d in DOMINIOS):
        # El resumen solo vale sobre las mismas entradas de dominio que leyó al escribirse
        firmas = {d: cache.firma(entradas[d]) for d in DOMINIOS}
        r, motivo, _ = cache.vigente(artefactos, ORQUESTADOR, entradas.get(ORQUESTADOR), firmas)
    else:
        r, motivo = None, "algún dominio no salió de la caché"
    resultados[ORQUESTADOR] = r or plantilla_orquestador(resultados, union, motivo)
    return resultados


# --- en vivo -------------------------------------------------------------------------------------------


def _agente(c: Corrida, dominio: str) -> Agent[None, SalidaDominio]:
    return Agent(
        c.modelo,
        name=f"agente_{dominio}",
        instructions=prompts.instrucciones(dominio),
        output_type=proveedor.tipo_salida(c.modelo, SalidaDominio),
        model_settings=c.ajustes,
        capabilities=[Freno(c.presupuesto)],
    )


def _motivo_de(e: Exception) -> str:
    """Por qué falló una corrida de agente, sin datos sensibles: las keys nunca viajan en estos errores."""
    if isinstance(e, TopeDeCorrida | RechazoDelModelo):
        return str(e)
    if isinstance(e, ContentFilterError):
        return f"el modelo rechazó la solicitud ({e})"
    if isinstance(e, UsageLimitExceeded):
        return f"límite de uso del agente: {e}"
    return f"error del modelo ({type(e).__name__}): {str(e)[:MAX_ERROR]}"


async def _con_reintento(
    c: Corrida,
    dominio: str,
    agente: Agent[None, SalidaDominio],
    primer_mensaje: str,
    respuestas_de: Callable[[list[ModelMessage]], Respuestas],
    **run: Any,
) -> tuple[ResultadoDominio | None, str | None, list[Rechazo], Respuestas, int]:
    """El primer intento y, si el validador rechaza cifras, el reintento sobre el mismo historial.

    Devuelve el resultado en vivo o, si cae, el motivo, las cifras rechazadas y los intentos hechos."""
    c.presupuesto.dominio = dominio
    historial: list[ModelMessage] | None = None
    uso: RunUsage | None = None
    rechazadas: list[Rechazo] = []
    respuestas: Respuestas = []
    for intento in range(2):
        if c.presupuesto.agotado():
            return (
                None,
                f"la corrida llegó al tope de USD {c.presupuesto.tope_usd:g}",
                rechazadas,
                respuestas,
                intento,
            )
        mensaje = primer_mensaje if intento == 0 else prompts.reintento(rechazadas)
        try:
            r = await agente.run(mensaje, message_history=historial, usage=uso, usage_limits=LIMITES, **run)
        except Exception as e:  # cualquier falla del modelo deja el dominio en su plantilla
            return None, _motivo_de(e), rechazadas, respuestas, intento
        historial, uso = r.all_messages(), r.usage
        respuestas = respuestas_de(historial)
        v = _validar(r.output.insights, respuestas, dominio)
        if v.valida:
            resultado = ResultadoDominio(
                origen="vivo",
                insights=r.output.insights,
                validacion=v,
                rechazadas_previas=rechazadas,
                reintentos=intento,
                modelo=c.nombre,
                llamadas=cache.unicas(ll for ll, _ in respuestas),
            )
            return resultado, None, rechazadas, respuestas, intento
        rechazadas = v.rechazadas
    return None, "el validador rechazó cifras también en el reintento", rechazadas, respuestas, 1


async def _dominio(
    c: Corrida, artefactos: Path, dominio: str, toolset: MCPToolset
) -> tuple[ResultadoDominio, Respuestas]:
    nombres = DOMINIOS[dominio]
    filtrado = toolset.filtered(lambda ctx, td: td.name in nombres)

    def llamadas(mensajes: list[ModelMessage]) -> Respuestas:
        return [
            (Llamada(herramienta=n, argumentos=a), r)
            for n, a, r in validador.respuestas_de(mensajes, nombres)
        ]

    resultado, motivo, rechazadas, respuestas, intentos = await _con_reintento(
        c, dominio, _agente(c, dominio), prompts.USUARIO, llamadas, toolsets=[filtrado]
    )
    if resultado is None:
        resultado, respuestas = plantilla_dominio(artefactos, dominio, motivo)
        resultado.rechazadas_previas, resultado.reintentos = rechazadas, intentos
    resultado.gasto = c.presupuesto.por_dominio.get(dominio)
    return resultado, respuestas


async def _orquestador(
    c: Corrida, por_dominio: dict[str, ResultadoDominio], union: Respuestas
) -> ResultadoDominio:
    entrada = json.dumps(
        {d: [i.model_dump() for i in r.insights] for d, r in por_dominio.items()},
        ensure_ascii=False,
        indent=1,
    )
    resultado, motivo, rechazadas, _, intentos = await _con_reintento(
        c, ORQUESTADOR, _agente(c, ORQUESTADOR), prompts.usuario_orquestador(entrada), lambda _: union
    )
    if resultado is None:
        resultado = plantilla_orquestador(por_dominio, union, motivo)
        resultado.rechazadas_previas, resultado.reintentos = rechazadas, intentos
    resultado.gasto = c.presupuesto.por_dominio.get(ORQUESTADOR)
    return resultado


async def en_vivo(
    artefactos: Path, c: Corrida, progreso: Callable[[str], None] | None = None
) -> tuple[dict[str, ResultadoDominio], dict[str, Respuestas]]:
    """Los cuatro agentes en vivo; devuelve los resultados y las respuestas de cada dominio para la caché."""
    toolset = MCPToolset(servidor.crear(artefactos))
    resultados: dict[str, ResultadoDominio] = {}
    respuestas: dict[str, Respuestas] = {}
    async with toolset:
        for dominio in DOMINIOS:
            resultados[dominio], respuestas[dominio] = await _dominio(c, artefactos, dominio, toolset)
            if progreso is not None:
                progreso(_avance(dominio, resultados[dominio]))
    union = [x for d in DOMINIOS for x in respuestas[d]]
    resultados[ORQUESTADOR] = await _orquestador(c, resultados, union)
    respuestas[ORQUESTADOR] = union
    if progreso is not None:
        progreso(_avance(ORQUESTADOR, resultados[ORQUESTADOR]))
    return resultados, respuestas


def _avance(dominio: str, r: ResultadoDominio) -> str:
    costo = "" if r.gasto is None else f" · USD {r.gasto.costo_usd:.4f}"
    motivo = "" if r.motivo is None else f" · {r.motivo}"
    return f"{dominio}: {r.origen}{costo}{motivo}"
