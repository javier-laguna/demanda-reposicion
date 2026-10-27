"""Las páginas del tablero: un script por página, que `principal.py` registra con `st.Page`.

Aquí viven las piezas que comparten: los formatos de número y el bloque con los insights de un dominio.
"""

from collections.abc import Mapping
from typing import Any, Literal

import streamlit as st

from demanda.app import datos

Color = Literal["green", "blue", "orange", "gray"]
# Cómo se muestra el origen de los insights: etiqueta y color de la insignia
ORIGENES: dict[str, tuple[str, Color]] = {
    "vivo": ("en vivo", "green"),
    "cache": ("caché", "blue"),
    "plantilla": ("plantilla", "orange"),
}
AGENTES = {
    "datos": "agente de datos",
    "pronostico": "agente de pronóstico",
    "causal": "agente causal",
    "orquestador": "orquestador",
}
BUEN_FIN = (
    "El Buen Fin ocurre una sola vez en los datos: su efecto, su error y su costo no se pueden validar fuera "
    "de muestra."
)


def pct(x: float | None, decimales: int = 2, signo: bool = False) -> str:
    return "—" if x is None else f"{x:{'+' if signo else ''}.{decimales}f} %"


def valor_p(x: float | None) -> str:
    return "—" if x is None else f"{x:.3g}"


def si_no(x: bool | None) -> str:
    return "—" if x is None else ("sí" if x else "no")


def _texto(s: str) -> str:
    # En el markdown de Streamlit, «$» abre una fórmula
    return s.replace("$", r"\$")


def insignia_origen(origen: str) -> None:
    etiqueta, color = ORIGENES.get(origen, (origen, "gray"))
    st.badge(f"origen: {etiqueta}", color=color)


def tarjeta(insight: Mapping[str, Any]) -> None:
    """Un insight: título, hallazgo, acción, confianza y las cifras que lo sostienen."""
    with st.container(border=True):
        st.markdown(f"**{_texto(insight['titulo'])}**")
        st.markdown(_texto(insight["hallazgo"]))
        st.markdown(f"**Acción.** {_texto(insight['accion'])}")
        evidencia = "  \n".join(
            f"`{e['cifra']}` {_texto(e['que_mide'])} · {e['herramienta']}" for e in insight["evidencia"]
        )
        st.caption(f"Confianza {insight['confianza']}.  \n{evidencia}")


def insights(dominio: str) -> None:
    """Los insights de un dominio en un desplegable, con su origen y el resultado del validador."""
    r = datos.artefacto("insights/insights.json")
    if r is None:
        return
    d = r["dominios"][dominio]
    v = d["validacion"]
    etiqueta = ORIGENES.get(d["origen"], (d["origen"], "gray"))[0]
    estado = "validador ok" if v["valida"] else f"{len(v['rechazadas'])} cifras rechazadas"
    titulo = f"Lo que interpreta el {AGENTES[dominio]} · {etiqueta} · {estado}"
    with st.expander(titulo, icon=":material/psychology:"):
        insignia_origen(d["origen"])
        if d["motivo"]:
            st.caption(f"Por qué no salió en vivo: {d['motivo']}")
        st.caption(
            f"El validador revisó {v['cifras']} cifras contra las herramientas MCP. Son los insights de la "
            "última corrida de `make insights`."
        )
        for insight in d["insights"]:
            tarjeta(insight)
