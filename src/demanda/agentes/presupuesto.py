"""El freno de gasto propio de la capa de agentes. El límite duro va en la consola del proveedor; este lo
complementa en tres niveles:

- **por agente:** `LIMITES`, los `UsageLimits` de Pydantic AI (8 solicitudes y 150 mil tokens), que se
  cuentan juntos para el primer intento y el reintento;
- **por corrida:** `Presupuesto` estima el costo de cada respuesta con sus tokens y la tabla de precios de
  `proveedor.py`, y `Freno` detiene la solicitud siguiente cuando la corrida llega a `LLM_TOPE_CORRIDA_USD`;
- **acumulado:** `gasto.json` suma el costo estimado de todas las corridas en vivo de la máquina, y el modo
  vivo no arranca si ya llegó a `LLM_PRESUPUESTO_USD`.

`Freno` también convierte un rechazo del modelo (`stop_reason: refusal`, que Pydantic AI mapea a
`finish_reason='content_filter'`) en `RechazoDelModelo`: cuenta como validación fallida y el dominio cae a
la plantilla.
"""

import json
from dataclasses import KW_ONLY, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.usage import RequestUsage

from demanda.agentes.modelos import Gasto

LIMITES = UsageLimits(request_limit=8, total_tokens_limit=150_000)


class TopeDeCorrida(RuntimeError):
    """La corrida llegó al tope de gasto: no se manda la solicitud siguiente."""


class RechazoDelModelo(RuntimeError):
    """El modelo rechazó la solicitud por seguridad."""


@dataclass
class Presupuesto:
    """El gasto estimado de una corrida, total y por dominio."""

    tope_usd: float
    precio: tuple[float, float]  # USD por millón de tokens de entrada y de salida
    gastado_usd: float = 0.0
    dominio: str = ""
    por_dominio: dict[str, Gasto] = field(default_factory=dict)

    def costo(self, entrada: int, salida: int) -> float:
        return (entrada * self.precio[0] + salida * self.precio[1]) / 1e6

    def registrar(self, uso: RequestUsage) -> None:
        costo = self.costo(uso.input_tokens, uso.output_tokens)
        self.gastado_usd += costo
        g = self.por_dominio.setdefault(self.dominio, Gasto())
        g.solicitudes += 1
        g.tokens_entrada += uso.input_tokens
        g.tokens_salida += uso.output_tokens
        g.costo_usd = round(g.costo_usd + costo, 6)

    def agotado(self) -> bool:
        return self.gastado_usd >= self.tope_usd

    def total(self) -> Gasto:
        gastos = self.por_dominio.values()
        return Gasto(
            solicitudes=sum(g.solicitudes for g in gastos),
            tokens_entrada=sum(g.tokens_entrada for g in gastos),
            tokens_salida=sum(g.tokens_salida for g in gastos),
            costo_usd=round(self.gastado_usd, 6),
        )


@dataclass
class Freno(AbstractCapability[Any]):
    """Registra el costo de cada respuesta, corta antes de pasar el tope y convierte los rechazos en error."""

    presupuesto: Presupuesto
    _: KW_ONLY
    id: str | None = "freno_de_gasto"

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        p = self.presupuesto
        if p.agotado():
            raise TopeDeCorrida(
                f"la corrida llegó al tope de USD {p.tope_usd:g} (gastado USD {p.gastado_usd:.4f})"
            )
        return request_context

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        self.presupuesto.registrar(response.usage)
        if response.finish_reason == "content_filter":
            detalle = (response.provider_details or {}).get("refusal_category") or "sin categoría"
            raise RechazoDelModelo(f"el modelo rechazó la solicitud ({detalle})")
        return response


def leer_gasto(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        return {"acumulado_usd": 0.0, "corridas": []}
    return json.loads(ruta.read_text())


def registrar_corrida(ruta: Path, modelo: str, presupuesto: Presupuesto) -> dict[str, Any]:
    """Suma la corrida a `gasto.json` y devuelve el registro actualizado."""
    gasto = leer_gasto(ruta)
    total = presupuesto.total()
    gasto["corridas"].append(
        {
            "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
            "modelo": modelo,
            "precio_usd_por_millon": {"entrada": presupuesto.precio[0], "salida": presupuesto.precio[1]},
            "tope_usd": presupuesto.tope_usd,
            **total.model_dump(),
            "por_dominio": {d: g.model_dump() for d, g in presupuesto.por_dominio.items()},
        }
    )
    gasto["acumulado_usd"] = round(gasto.get("acumulado_usd", 0.0) + total.costo_usd, 6)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(gasto, indent=2, ensure_ascii=False) + "\n")
    return gasto
