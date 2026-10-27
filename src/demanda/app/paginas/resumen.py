"""Resumen: los indicadores clave y lo que concluye el orquestador, tomados tal cual de los artefactos."""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos
from demanda.app.paginas import AGENTES, insignia_origen, pct, tarjeta, valor_p
from demanda.seleccion.campeon import campeon_y_retador


def diferencia_dm(pruebas: list[dict[str, Any]], a: str, b: str) -> tuple[float, float | None] | None:
    """La Diebold-Mariano global de `a` contra `b`: la diferencia de WAPE (pp, a − b) y su p. Si
    `seleccion.json` la trae en el otro orden, la diferencia cambia de signo; si no la trae, `None`."""
    for p in pruebas:
        if p["segmento"] == "global" and {p["a"], p["b"]} == {a, b}:
            signo = 1 if p["a"] == a else -1
            return signo * p["diferencia_wape_pp"], p["p_valor"]
    return None


st.title("Pronóstico de demanda para reposición")
st.caption(
    "Retail con tiendas en varias regiones de México: pronóstico de `units_sold` por tienda × categoría a "
    "7 días contra el sistema de reposición actual, una regla de decisión, un monitor de drift y un "
    "experimento natural con las promociones."
)

st.subheader("Indicadores clave")
seleccion = datos.artefacto("modelos/seleccion.json")
impacto = datos.artefacto("decision/impacto.json")
promociones = datos.artefacto("causal/promociones.json")
columnas = st.columns(4)

if seleccion is not None:
    campeon, retador = campeon_y_retador(seleccion)
    m = seleccion["modelos"]
    dm = diferencia_dm(seleccion["pruebas_dm"], retador, campeon)
    columnas[0].metric(
        "WAPE del campeón",
        pct(m[campeon]["wape_semanal_global_pct"]),
        help=f"WAPE semanal global. En semanas normales: {pct(m[campeon]['wape_semanal_resto_pct'])}.",
        border=True,
    )
    columnas[0].caption(f"`{campeon}`, semanal, en los cortes del backtest.")
    columnas[1].metric(
        "WAPE del retador",
        pct(m[retador]["wape_semanal_global_pct"]),
        delta=None if dm is None else f"{dm[0]:+.2f} pp contra el campeón",
        delta_color="inverse",
        help=f"WAPE semanal global. En semanas normales: {pct(m[retador]['wape_semanal_resto_pct'])}.",
        border=True,
    )
    columnas[1].caption(
        f"`{retador}`; Diebold-Mariano por corte: p = {valor_p(dm[1])}."
        if dm is not None
        else f"`{retador}`; `seleccion.json` no trae su Diebold-Mariano global contra el campeón."
    )

if impacto is not None:
    # La prueba de E5 del newsvendor sobre el pronóstico del campeón contra reponer a su pronóstico puntual
    # (solo en ese orden: `diferencia_pct` es relativa al costo de `b`, así que no se invierte)
    par = (f"newsvendor_{impacto['campeon']}", f"puntual_{impacto['campeon']}", "global")
    prueba = next((p for p in impacto["pruebas_costo"] if (p["a"], p["b"], p["segmento"]) == par), None)
    if prueba is None:
        columnas[2].warning(
            f"`decision/impacto.json` no trae la prueba de costo de {par[0]} contra {par[1]} en el global.",
            icon=":material/warning:",
        )
    else:
        columnas[2].metric(
            "Ahorro del newsvendor",
            pct(-prueba["diferencia_pct"], 1),
            help=f"Costo de {prueba['a']} contra {prueba['b']}: {pct(prueba['diferencia_pct'], signo=True)} "
            f"(Diebold-Mariano p = {valor_p(prueba['p_valor'])}).",
            border=True,
        )
    columnas[2].caption("Con márgenes y costos de mantener supuestos.")

if promociones is not None:
    e = promociones["efecto_principal"]
    bajo, alto = e["ic95_pct"]
    columnas[3].metric(
        "Efecto de la promoción",
        pct(e["efecto_pct"], signo=True),
        help=f"Sobre las unidades vendidas; p = {valor_p(e['p_valor'])} con errores agrupados por tienda.",
        border=True,
    )
    columnas[3].caption(f"IC95 de {pct(bajo, signo=True)} a {pct(alto, signo=True)}.")

st.subheader("Lo que concluye el orquestador")
insights = datos.artefacto("insights/insights.json")
if insights is not None:
    orquestador = insights["dominios"]["orquestador"]
    v = insights["validacion"]
    insignia_origen(orquestador["origen"])
    st.caption(
        f"Modo {insights['modo']} (pedido: {insights['modo_solicitado']}) · modelo {insights['modelo']} · "
        f"prompts {insights['prompt_version']}" + (f" · {insights['aviso']}" if insights["aviso"] else "")
    )
    if v["todas_validas"]:
        st.success(
            f"El validador revisó {v['cifras']} cifras de todos los dominios contra las herramientas MCP: "
            "ninguna rechazada.",
            icon=":material/verified:",
        )
    else:
        st.error(f"El validador rechazó {v['rechazadas']} de {v['cifras']} cifras.", icon=":material/error:")
    izquierda, derecha = st.columns(2)
    for i, insight in enumerate(orquestador["insights"]):
        with izquierda if i % 2 == 0 else derecha:
            tarjeta(insight)

    with st.expander("Origen de cada dominio"):
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "dominio": AGENTES.get(nombre, nombre),
                        "origen": d["origen"],
                        "insights": len(d["insights"]),
                        "cifras revisadas": d["validacion"]["cifras"],
                        "rechazadas": len(d["validacion"]["rechazadas"]),
                        "reintentos": d["reintentos"],
                        "motivo": d["motivo"] or "",
                    }
                    for nombre, d in insights["dominios"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption("Son los insights de la última corrida de `make insights`.")
