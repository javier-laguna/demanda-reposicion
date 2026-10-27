"""Modelos (E3 y E4): la comparación del backtest por segmento y evento, la regla de campeón y la
cobertura de los intervalos."""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos, graficas
from demanda.app.paginas import BUEN_FIN, insights, pct, si_no, valor_p
from demanda.seleccion.campeon import CRITERIOS, campeon_y_retador

SEGMENTOS = {
    "evento": "evento",
    "formato": "formato",
    "region": "región",
    "nse": "NSE",
    "categoria": "categoría",
}

st.title("Modelos")
insights("pronostico")


def tabla_global(
    metricas: dict[str, Any], seleccion: dict[str, Any], campeon: str, retador: str
) -> pl.DataFrame:
    """Una fila por modelo: sus métricas globales de `metricas.json` y las de la regla de `seleccion.json`."""
    valores = {
        (f["modelo"], f"{f['granularidad']}_{f['metrica']}"): f["valor"]
        for f in metricas["segmentos"]
        if f["segmento"] == "global"
    }
    papeles = {campeon: "campeón", retador: "retador"}
    filas = []
    for modelo in metricas["protocolo"]["modelos"]:
        regla = seleccion["modelos"].get(modelo, {})
        filas.append(
            {
                "modelo": modelo,
                "papel": papeles.get(modelo, ""),
                "WAPE semanal (%)": valores.get((modelo, "semanal_wape_pct")),
                "en semanas normales (%)": regla.get("wape_semanal_resto_pct"),
                "sesgo semanal (%)": valores.get((modelo, "semanal_sesgo_pct")),
                "|sesgo| máx. por subgrupo (%)": regla.get("sesgo_semanal_max_abs_pct"),
                "WAPE diario (%)": valores.get((modelo, "diaria_wape_pct")),
                "cobertura P10–P90 (%)": valores.get((modelo, "diaria_cobertura_80_pct")),
            }
        )
    return pl.DataFrame(filas)


def comparacion(metricas: dict[str, Any], seleccion: dict[str, Any], campeon: str, retador: str) -> None:
    st.subheader("Comparación de los modelos")
    protocolo = metricas["protocolo"]
    st.caption(
        f"Backtest de origen rodante: {len(protocolo['origenes'])} cortes semanales, horizonte de "
        f"{protocolo['horizonte']} días. La métrica principal es el WAPE de la suma semanal por serie y "
        "corte; lo diario es diagnóstico."
    )
    st.dataframe(
        tabla_global(metricas, seleccion, campeon, retador),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in (
                "WAPE semanal (%)",
                "en semanas normales (%)",
                "sesgo semanal (%)",
                "|sesgo| máx. por subgrupo (%)",
                "WAPE diario (%)",
                "cobertura P10–P90 (%)",
            )
        },
    )
    semanal = metricas["cobertura"]["semanal"]
    st.caption(
        f"Las métricas semanales usan {semanal['pares_completos']:,} de {semanal['pares_total']:,} pares "
        f"serie × corte ({pct(semanal['pct'])}): los que tienen los {protocolo['horizonte']} días observados."
    )
    disponibles = [s for s in SEGMENTOS if any(f["segmento"] == s for f in metricas["segmentos"])]
    segmento = st.radio(
        "Segmento", disponibles, format_func=lambda s: SEGMENTOS[s], horizontal=True, key="segmento_modelos"
    )
    if segmento is None:
        return
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("**WAPE semanal**")
        st.altair_chart(graficas.metrica_por_segmento(metricas, segmento), width="stretch")
    with derecha:
        st.markdown("**Sesgo semanal**")
        st.altair_chart(graficas.metrica_por_segmento(metricas, segmento, "sesgo_pct"), width="stretch")
    if segmento == "evento":
        st.caption(
            "`post_buen_fin` es la semana siguiente al Buen Fin (la señal del sistema actual promedia el "
            "pico) y `resto`, las semanas normales."
        )
        st.info(BUEN_FIN, icon=":material/event_busy:")


def regla(seleccion: dict[str, Any], campeon: str) -> None:
    st.subheader("Regla de campeón")
    r = seleccion["regla"]
    st.markdown(
        f"Se fijó antes de ver los resultados. Un retador reemplaza a `{r['incumbente']}` solo si cumple "
        f"las cuatro condiciones (α = {r['alfa']}, margen en semanas normales {r['margen_resto_pp']} pp, "
        f"tope de sesgo {r['sesgo_max_pct']} %):"
    )
    for nombre, texto in r["criterios"].items():
        st.markdown(f"- **{nombre}**: {texto}")
    filas = []
    for modelo, m in seleccion["modelos"].items():
        if m["criterios"] is None:
            continue
        celdas = {
            n: f"{si_no(c['cumple'])} · {c['valor']} vs {c['umbral']}" for n, c in m["criterios"].items()
        }
        filas.append(
            {"modelo": modelo, **{n: celdas.get(n, "—") for n in CRITERIOS}, "pasa": si_no(m["pasa"])}
        )
    st.dataframe(pl.DataFrame(filas), hide_index=True, width="stretch")
    st.success(f"Campeón: **{campeon}**. {seleccion['justificacion']}", icon=":material/emoji_events:")
    with st.expander("Pruebas de Diebold-Mariano por corte"):
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "a": p["a"],
                        "b": p["b"],
                        "segmento": p["segmento"],
                        "cortes": p["n_cortes"],
                        "Δ WAPE (pp)": p["diferencia_wape_pp"],
                        "p": valor_p(p["p_valor"]),
                        "a favor de": p["a_favor_de"],
                    }
                    for p in seleccion["pruebas_dm"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("d = |E_a| − |E_b| por corte: negativa favorece a `a`.")
    with st.expander("Sensibilidad de la regla"):
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "margen en semanas normales (pp)": f["margen_resto_pp"],
                        "α": f["alfa"],
                        "pasan": ", ".join(f["pasan"]) or "ninguno",
                        "campeón": f["campeon"],
                    }
                    for f in seleccion["sensibilidad"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def cobertura(metricas: dict[str, Any]) -> None:
    st.subheader("Cobertura de los intervalos")
    st.altair_chart(graficas.cobertura_intervalos(metricas), width="stretch")
    st.caption(
        "Proporción de días reales dentro del P10–P90 de cada modelo con intervalo; el nominal es 80 % "
        "(línea). Los modelos puntuales no aparecen."
    )


fuentes = datos.cargar("backtest/metricas.json", "modelos/seleccion.json")
if fuentes is not None:
    metricas, seleccion = fuentes
    campeon, retador = campeon_y_retador(seleccion)
    comparacion(metricas, seleccion, campeon, retador)
    regla(seleccion, campeon)
    cobertura(metricas)
