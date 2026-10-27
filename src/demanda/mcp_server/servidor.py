"""El servidor MCP de solo lectura: expone las siete funciones de `herramientas.py` sobre un directorio de
artefactos.

- Los agentes de `make insights` lo usan en proceso, sin subproceso (`MCPToolset(crear(...))`).
- Los clientes externos (Claude Desktop, Claude Code u otro agente) lo levantan con `demanda mcp`: por
  stdio, o por streamable HTTP con `--http`.

Ninguna herramienta acepta rutas: el directorio se fija al crear el servidor. Los errores de dominio
(argumento fuera del catálogo, artefacto faltante) llegan al cliente como `ToolError` con su mensaje.
"""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from demanda.mcp_server import herramientas
from demanda.mcp_server.herramientas import ArgumentoInvalido, ArtefactoFaltante

NOMBRE = "demanda"
PUERTO = 8765
INSTRUCCIONES = (
    "Herramientas de solo lectura sobre los artefactos de un pipeline de pronóstico de demanda de retail "
    "(80 tiendas × 6 categorías): calidad de datos, EDA, métricas del backtest y regla de campeón, "
    "pronósticos por serie, monitor de drift, impacto de la regla de reposición y efecto de las promociones. "
    "Las cifras ya están calculadas; no hay herramientas que escriban."
)
SOLO_LECTURA = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _errores_como_tool_error(f: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(f)
    def envuelta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return f(*args, **kwargs)
        except (ArtefactoFaltante, ArgumentoInvalido) as e:
            raise ToolError(str(e)) from e

    return envuelta


def crear(artefactos: Path) -> FastMCP:
    """El servidor con las siete herramientas sobre `artefactos`."""
    mcp = FastMCP(NOMBRE, instructions=INSTRUCCIONES)

    def registrar(f: Callable[..., dict[str, Any]]) -> None:
        mcp.tool(_errores_como_tool_error(f), annotations=SOLO_LECTURA)

    @registrar
    def get_data_quality() -> dict[str, Any]:
        """Calidad de los datos (E1): contratos, malla de tienda × categoría × día con sus huecos, conteos
        inconsistentes, nulos por columna y la fuga de información de replenishment_signal."""
        return herramientas.get_data_quality(artefactos)

    @registrar
    def get_eda_summary() -> dict[str, Any]:
        """Resumen del EDA (E2): nivel y estacionalidad de la demanda, efectos de calendario, promociones,
        tiendas, desempeño del sistema de reposición actual e implicaciones para el modelado."""
        return herramientas.get_eda_summary(artefactos)

    @registrar
    def get_metrics(segmento: str = "global") -> dict[str, Any]:
        """Métricas del backtest (E3) en un segmento: WAPE y sesgo semanales, WAPE diario y cobertura del
        intervalo P10–P90 por modelo y nivel, más la regla de campeón (E4) con sus criterios, pruebas
        Diebold-Mariano y justificación. segmento: global, evento, formato, region, nse, categoria, corte
        u horizonte."""
        return herramientas.get_metrics(artefactos, segmento)

    @registrar
    def get_forecast(store_id: str, category: str, origen: str | None = None) -> dict[str, Any]:
        """Pronóstico diario del campeón y del retador, con P10–P90 y el real, para una tienda (p. ej.
        STR_001) y categoría en un corte del backtest (fecha ISO del origen; el último por omisión)."""
        return herramientas.get_forecast(artefactos, store_id, category, origen)

    @registrar
    def get_drift_alerts(semana: str | None = None) -> dict[str, Any]:
        """Monitor de drift (E5): resumen por capa, semanas con alertas inesperadas y sus recomendaciones,
        vigilancia del retador y validación con drift inyectado. Con semana (fecha ISO del origen T), el
        detalle de las alertas de esa semana."""
        return herramientas.get_drift_alerts(artefactos, semana)

    @registrar
    def get_decision_impact() -> dict[str, Any]:
        """Regla de decisión (E5): políticas de reposición, impacto global y por evento (costo, quiebre,
        sobreinventario, fill rate), pruebas sobre el costo, sensibilidad y conclusión. Los costos son
        supuestos ilustrativos."""
        return herramientas.get_decision_impact(artefactos)

    @registrar
    def get_promo_effect() -> dict[str, Any]:
        """Experimento de promociones (E6): balance de la aleatorización, efecto principal, canales,
        robustez, heterogeneidad, dinámica, potencia, traducción a negocio y conclusión."""
        return herramientas.get_promo_effect(artefactos)

    return mcp


def servir(artefactos: Path, http: bool = False, host: str = "127.0.0.1", puerto: int = PUERTO) -> None:
    """Levanta el servidor: por stdio (lo que esperan Claude Desktop y Claude Code) o por streamable HTTP."""
    mcp = crear(artefactos)
    if http:
        mcp.run(transport="http", host=host, port=puerto, show_banner=False)
    else:
        # En stdio, stdout es el canal del protocolo: nada más puede escribir ahí
        mcp.run(transport="stdio", show_banner=False)
