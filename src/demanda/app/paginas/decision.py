"""Decisión y drift: las políticas de reposición con su costo y el monitor semana a semana (E5)."""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos, graficas
from demanda.app.paginas import BUEN_FIN, insights, si_no, valor_p

st.title("Decisión y drift")
insights("pronostico")


def politicas(impacto: dict[str, Any]) -> None:
    st.subheader("Regla de decisión: newsvendor")
    d = impacto["definiciones"]
    st.markdown(f"Nivel objetivo semanal **S = Σŷ × q**, donde q es el {d['q']}.")
    st.info(
        "Los costos son supuestos: márgenes y costos de mantener ilustrativos (`decision/supuestos.py`); el "
        "precio sale de los datos. Las cifras en pesos sirven para comparar políticas, no como costo real.",
        icon=":material/info:",
    )
    st.dataframe(
        pl.DataFrame([{"política": p, "definición": texto} for p, texto in impacto["politicas"].items()]),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        pl.DataFrame(
            [
                {
                    "política": f["politica"],
                    "costo supuesto (M$)": f["costo"] / 1e6,
                    "sobreinventario (%)": f["sobreinventario_pct"],
                    "quiebre (%)": f["quiebre_pct"],
                    "fill rate (%)": f["fill_rate_pct"],
                    "nivel de servicio (%)": f["nivel_servicio_pct"],
                    "objetivo (%)": f["fc_objetivo_pct"],
                }
                for f in impacto["impacto"]
                if f["segmento"] == "global"
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "costo supuesto (M$)": st.column_config.NumberColumn(format="%.1f"),
            **{
                c: st.column_config.NumberColumn(format="%.2f")
                for c in (
                    "sobreinventario (%)",
                    "quiebre (%)",
                    "fill rate (%)",
                    "nivel de servicio (%)",
                    "objetivo (%)",
                )
            },
        },
    )
    st.caption(
        f"Quiebre y sobreinventario en % de la demanda; nivel de servicio: {d['nivel_servicio_pct']}; "
        f"objetivo: {d['fc_objetivo_pct']}."
    )
    st.markdown("**Costo por política y evento**")
    st.altair_chart(graficas.costo_por_politica(impacto), width="stretch")
    st.info(BUEN_FIN, icon=":material/event_busy:")
    st.markdown(impacto["conclusion"])
    with st.expander("Pruebas de Diebold-Mariano sobre el costo por corte"):
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "a": p["a"],
                        "b": p["b"],
                        "segmento": p["segmento"],
                        "cortes": p["n_cortes"],
                        "Δ costo (%)": p["diferencia_pct"],
                        "p": valor_p(p["p_valor"]),
                        "a favor de": p["a_favor_de"],
                    }
                    for p in impacto["pruebas_costo"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Supuestos de costo por categoría"):
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "categoría": c,
                        "margen bruto (supuesto)": s["margen_bruto"],
                        "costo de mantener por semana (supuesto)": s["mantener_por_semana"],
                        "fracción crítica": s["fraccion_critica"],
                        "nota": s["nota"],
                    }
                    for c, s in impacto["supuestos"]["costos"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def sensibilidad(impacto: dict[str, Any]) -> None:
    st.subheader("Sensibilidad a la fracción crítica")
    s = impacto["sensibilidad"]
    st.altair_chart(graficas.sensibilidad_costo(impacto), width="stretch")
    st.caption(
        f"Misma fracción crítica en todas las categorías. El orden de las políticas por costo "
        f"{'se mantiene' if s['orden_se_mantiene'] else 'cambia'} en las {len(s['fracciones'])} fracciones; "
        f"con las fracciones supuestas, de menor a mayor costo: {', '.join(s['orden_base'])}."
    )


def monitor(m: dict[str, Any]) -> None:
    st.subheader("Monitor de drift semana a semana")
    conf = m["configuracion"]
    st.caption(
        f"Referencia: {conf['referencia']}. Segmentos: {conf['segmentos']}. Se marcan como esperadas: "
        f"{conf['esperadas_por_calendario']}."
    )
    for capa in graficas.CAPAS:
        st.altair_chart(graficas.alertas_por_semana(m, capa), width="stretch")
    st.info(
        "Limitación conocida del monitor, no una falla: el chequeo de nivel de demanda compara cada semana "
        "contra su referencia reciente y confunde la estacionalidad anual con drift. Por eso las alertas de "
        "nivel de noviembre y enero salen inesperadas en casi todos los segmentos.",
        icon=":material/info:",
    )
    # Los conteos de las tablas salen de `resumen`, que ya los agrega; no se recuentan las alertas
    resumen = m["resumen"]
    st.dataframe(
        pl.DataFrame(
            [
                {"periodo": periodo, "capa": capa, **conteos}
                for periodo in ("calibracion", "backtest")
                for capa, conteos in resumen[periodo]["por_capa"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Alertas por chequeo"):
        st.dataframe(
            pl.DataFrame(
                [
                    {"periodo": periodo, "chequeo": chequeo, **conteos}
                    for periodo in ("calibracion", "backtest")
                    for chequeo, conteos in resumen[periodo]["por_chequeo"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Semanas con alertas inesperadas y sus recomendaciones"):
        filas = [
            {
                "semana (origen T)": s["origen"],
                "periodo": s["periodo"],
                "evento": s["evento"],
                "día especial": si_no(s["dia_especial"]),
                "inesperadas": s["inesperadas"],
                "esperadas": s["esperadas"],
                "recomendaciones": ", ".join(s["recomendaciones"]),
            }
            for s in resumen["por_semana"]
            if s["inesperadas"]
        ]
        if filas:
            st.dataframe(pl.DataFrame(filas), hide_index=True, width="stretch")
        else:
            st.caption("Ninguna semana tiene alertas inesperadas.")


def vigilancia(m: dict[str, Any]) -> None:
    st.subheader("Vigilancia del retador")
    v = m["vigilancia_retador"]
    final, minima = v["final"] or {}, v["minima"] or {}
    ultimo = v["por_corte"][-1] if v["por_corte"] else {}
    st.altair_chart(graficas.vigilancia_retador(m), width="stretch")
    c = st.columns(4)
    c[0].metric("DM acumulada al último corte", f"p = {valor_p(final.get('p_valor'))}", border=True)
    c[1].metric(
        "p mínima en el backtest",
        f"p = {valor_p(minima.get('p_valor'))}",
        help=f"En el corte {minima.get('corte', '—')}.",
        border=True,
    )
    c[2].metric("Umbral con Bonferroni", f"p < {valor_p(ultimo.get('umbral_p'))}", border=True)
    c[3].metric("Cortes necesarios estimados", str(final.get("cortes_necesarios") or "—"), border=True)
    st.caption(
        f"El retador {m['retador']} se compara con el campeón {m['campeon']} al cerrar cada corte. Coincide "
        f"con `seleccion.json`: {si_no(v['coincide_con_seleccion'])}."
    )


def validacion(m: dict[str, Any]) -> None:
    st.subheader("Validación con drift inyectado")
    val = m["validacion"]
    st.markdown(f"El monitor detecta todos los escenarios: **{si_no(val['todos_detectados'])}**.")
    st.dataframe(
        pl.DataFrame(
            [
                {
                    "escenario": e["escenario"],
                    "descripción": e["descripcion"],
                    "semana": e["origen"],
                    "periodo": e["periodo"],
                    "detectado": si_no(e["detectado"]),
                    "recomendación esperada": e["recomendacion_esperada"],
                    "recomendaciones": ", ".join(e["recomendaciones"]),
                    "alertas inesperadas": e["alertas_inesperadas"],
                }
                for e in val["escenarios"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


impacto = datos.artefacto("decision/impacto.json")
if impacto is not None:
    politicas(impacto)
    sensibilidad(impacto)
drift = datos.artefacto("drift/monitor.json")
if drift is not None:
    monitor(drift)
    vigilancia(drift)
    validacion(drift)
