"""El servidor FastMCP: con el cliente en memoria lista exactamente las siete herramientas y cada una
responde; por stdio, como lo levanta `make mcp`, un cliente MCP también las lista."""

import asyncio
import os
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from demanda.mcp_server import herramientas, servidor

ARGUMENTOS = {"get_forecast": {"store_id": "STR_001", "category": "Bebidas"}}


def test_lista_las_siete_herramientas_y_cada_una_responde(artefactos_e7):
    async def probar():
        async with Client(servidor.crear(artefactos_e7)) as cliente:
            tools = await cliente.list_tools()
            respuestas = {t.name: await cliente.call_tool(t.name, ARGUMENTOS.get(t.name, {})) for t in tools}
            return tools, respuestas

    tools, respuestas = asyncio.run(probar())
    assert sorted(t.name for t in tools) == sorted(herramientas.HERRAMIENTAS)
    assert all(t.annotations is not None and t.annotations.read_only_hint for t in tools)
    for nombre, r in respuestas.items():
        assert r.structured_content == herramientas.llamar(artefactos_e7, nombre, ARGUMENTOS.get(nombre))


def test_los_errores_de_dominio_llegan_con_su_mensaje(artefactos_e7):
    (artefactos_e7 / "causal" / "promociones.json").unlink()

    async def llamar(nombre, argumentos):
        async with Client(servidor.crear(artefactos_e7)) as cliente:
            await cliente.call_tool(nombre, argumentos)

    with pytest.raises(ToolError, match="segmento inválido"):
        asyncio.run(llamar("get_metrics", {"segmento": "pais"}))
    with pytest.raises(ToolError, match="make causal"):
        asyncio.run(llamar("get_promo_effect", {}))


def test_por_stdio_un_cliente_lista_las_herramientas(artefactos_e7, tmp_path):
    # El subproceso lee su config del entorno: con DEMANDA_ENV_FILE inexistente no ve el `.env` real
    sin_env = tmp_path / "no_existe.env"
    transporte = StdioTransport(
        command=sys.executable,
        args=["-m", "demanda.mcp_server"],
        env={**os.environ, "ARTIFACTS_DIR": str(artefactos_e7), "DEMANDA_ENV_FILE": str(sin_env)},
    )

    async def listar():
        async with Client(transporte) as cliente:
            tools = await cliente.list_tools()
            r = await cliente.call_tool("get_metrics", {"segmento": "evento"})
            return [t.name for t in tools], r.structured_content

    nombres, metricas = asyncio.run(listar())
    assert sorted(nombres) == sorted(herramientas.HERRAMIENTAS)
    assert metricas is not None and metricas["segmento"] == "evento"
