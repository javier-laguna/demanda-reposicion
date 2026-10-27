"""La salida de la capa de agentes: el `Insight`, su validación y el resultado de cada dominio.

Los tres dominios y las herramientas que puede leer cada uno viven aquí, junto con las llamadas por omisión
que usan las plantillas. El orquestador no tiene herramientas: resume los insights validados de los otros.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Herramienta = Literal[
    "get_data_quality",
    "get_eda_summary",
    "get_metrics",
    "get_forecast",
    "get_drift_alerts",
    "get_decision_impact",
    "get_promo_effect",
]
Origen = Literal["vivo", "cache", "plantilla"]

# `get_forecast` no la usa ningún agente: devuelve la serie de una tienda, y los insights se versionan
DOMINIOS: dict[str, tuple[str, ...]] = {
    "datos": ("get_data_quality", "get_eda_summary"),
    "pronostico": ("get_metrics", "get_drift_alerts", "get_decision_impact"),
    "causal": ("get_promo_effect",),
}
ORQUESTADOR = "orquestador"
ORDEN = (*DOMINIOS, ORQUESTADOR)


class Llamada(BaseModel):
    """Una llamada a una herramienta: la caché las guarda para repetirlas al validar."""

    herramienta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)

    def clave(self) -> tuple[str, str]:
        return self.herramienta, repr(sorted(self.argumentos.items()))


# Las que usan las plantillas y la huella de un dominio cuando no hay historial de una corrida en vivo
LLAMADAS_POR_OMISION: dict[str, list[Llamada]] = {
    dominio: [Llamada(herramienta=h) for h in herramientas] for dominio, herramientas in DOMINIOS.items()
}


class Evidencia(BaseModel):
    cifra: str = Field(
        description="la cifra tal como la devuelve la herramienta, con su unidad; p. ej. 20.34 %"
    )
    que_mide: str = Field(description="qué mide la cifra, en pocas palabras")
    herramienta: Herramienta = Field(description="la herramienta de la que sale la cifra")


class Insight(BaseModel):
    titulo: str = Field(description="título corto")
    hallazgo: str = Field(description="de una a tres oraciones: qué dicen los resultados")
    evidencia: list[Evidencia] = Field(description="las cifras que sostienen el hallazgo")
    accion: str = Field(description="qué proponer al negocio")
    confianza: Literal["alta", "media", "baja"]


class SalidaDominio(BaseModel):
    """Lo que devuelve cada agente."""

    insights: list[Insight] = Field(min_length=2, max_length=4, description="de 2 a 4 insights")


class Rechazo(BaseModel):
    """Una cifra que no aparece en las herramientas, o una herramienta fuera del dominio."""

    cifra: str
    campo: str
    insight: int
    motivo: str


class Validacion(BaseModel):
    valida: bool
    cifras: int = Field(description="cifras revisadas")
    rechazadas: list[Rechazo] = Field(default_factory=list)


class Gasto(BaseModel):
    """El costo estimado de un dominio en una corrida en vivo."""

    solicitudes: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    costo_usd: float = 0.0


class ResultadoDominio(BaseModel):
    origen: Origen
    insights: list[Insight]
    validacion: Validacion
    # Por qué el dominio no salió en vivo o de la caché, y las cifras que lo hicieron caer
    motivo: str | None = None
    rechazadas_previas: list[Rechazo] = Field(default_factory=list)
    reintentos: int = 0
    modelo: str | None = None
    llamadas: list[Llamada] = Field(default_factory=list)
    gasto: Gasto | None = None
