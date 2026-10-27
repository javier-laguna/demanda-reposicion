"""Supuestos de costo del newsvendor: margen bruto y costo de mantener por semana, por categoría.

Son ilustrativos, porque los datos no los traen, y se escriben en la salida. Los dos van como fracción del
precio medio de la categoría, que sí sale de los datos:
- quiebre: Cu = margen bruto × precio, la utilidad que se pierde;
- sobreinventario: Co = costo de mantener semanal × precio, con capital, espacio, merma y rebaja.

La fracción crítica es Cu / (Cu + Co) = margen / (margen + mantener): no depende del precio.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Costo:
    """Por unidad y como fracción del precio medio de la categoría."""

    margen: float
    mantener: float  # por semana
    nota: str = ""

    @property
    def fraccion_critica(self) -> float:
        return self.margen / (self.margen + self.mantener)


COSTOS = MappingProxyType(
    {
        "Abarrotes": Costo(0.20, 0.020, "incluye merma"),
        "Bebidas": Costo(0.25, 0.015),
        "Cuidado_Personal": Costo(0.30, 0.015),
        "Electronica": Costo(0.12, 0.020, "obsolescencia"),
        "Hogar": Costo(0.30, 0.020),
        "Ropa": Costo(0.45, 0.040, "rebaja de temporada"),
    }
)


@dataclass(frozen=True)
class Supuestos:
    """Los costos por categoría y las fracciones críticas de la sensibilidad, fijados de antemano."""

    costos: MappingProxyType[str, Costo] = field(default_factory=lambda: COSTOS)
    fracciones_sensibilidad: tuple[float, ...] = (0.80, 0.90, 0.95, 0.98)

    def fracciones(self) -> dict[str, float]:
        return {c: costo.fraccion_critica for c, costo in self.costos.items()}

    def a_json(self) -> dict[str, Any]:
        return {
            "costos": {
                c: {
                    "margen_bruto": costo.margen,
                    "mantener_por_semana": costo.mantener,
                    "fraccion_critica": round(costo.fraccion_critica, 4),
                    "nota": costo.nota,
                }
                for c, costo in self.costos.items()
            },
            "fracciones_sensibilidad": list(self.fracciones_sensibilidad),
        }
