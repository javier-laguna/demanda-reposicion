"""Datos (E1 y E2): contratos, malla, nulos, la fuga de `replenishment_signal` y los efectos del
calendario."""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos, graficas
from demanda.app.paginas import BUEN_FIN, insights, pct

st.title("Datos")
st.caption("Qué se encontró en los datos antes de modelar: lo que el diccionario no menciona o describe mal.")
insights("datos")


def contratos_y_malla(r: dict[str, Any]) -> None:
    st.subheader("Contratos y malla")
    d, malla = r["dimensiones"], r["malla"]
    ok = sum(estado == "ok" for estado in r["contratos"].values())
    c = st.columns(4)
    c[0].metric("Contratos que se cumplen", f"{ok} de {len(r['contratos'])}", border=True)
    c[1].metric("Series tienda × categoría", f"{d['series']:,}", border=True)
    c[2].metric("Días", f"{d['dias']:,}", help=f"Del {d['desde']} al {d['hasta']}.", border=True)
    c[3].metric(
        "Filas que faltan en la malla",
        f"{malla['faltantes']:,}",
        help=f"De {malla['esperadas']:,} esperadas. No se imputan: quedan en el panel con una bandera.",
        border=True,
    )
    st.caption(
        "Contratos: "
        + " · ".join(f"{tabla} {estado}" for tabla, estado in r["contratos"].items())
        + ". El panel es la malla completa, así un desfase por serie siempre equivale a días de calendario."
    )
    if malla["bloques"]:
        st.dataframe(
            pl.DataFrame(malla["bloques"]).rename({"store_id": "tienda", "dias": "días"}),
            hide_index=True,
            width="stretch",
        )


def nulos_y_conteos(r: dict[str, Any]) -> None:
    st.subheader("Nulos y conteos")
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        st.altair_chart(graficas.nulos(r), width="stretch")
        st.caption("Ningún nulo se imputa: las filas sin `units_sold` no entran a entrenar ni a evaluar.")
    with derecha:
        conteos = r["conteos"]
        st.metric(
            "Filas con conteos inconsistentes",
            pct(conteos["pct_inconsistentes"]),
            help=f"{conteos['inconsistentes']:,} de {conteos['filas_evaluables']:,} filas evaluables.",
            border=True,
        )
        st.markdown(
            f"En esas filas `cash_transactions + card_transactions < total_transactions`, y `avg_ticket` se "
            f"calcula con efectivo más tarjeta, lo que sobreestima el ticket "
            f"{pct(conteos['avg_ticket']['sobreestimacion_media_pct_en_inconsistentes'], 1)} en promedio."
        )
        st.caption(
            "Uniformidad: "
            + " · ".join(
                f"por {k} de {pct(v['min'])} a {pct(v['max'])}" for k, v in conteos["uniformidad_pct"].items()
            )
        )


def fuga(r: dict[str, Any], eda: dict[str, Any] | None) -> None:
    st.subheader("La fuga de `replenishment_signal`")
    f = r["fuga_replenishment_signal"]
    v = f["mejor_ventana"]
    st.markdown(
        f"La señal fechada en t es ≈ {f['factor']} × la media de `units_sold` de t{v['desde']:+d} a "
        f"t{v['hasta']:+d} (correlación intraserie {v['corr']}): ya contiene las ventas de t y mira "
        f"{v['hasta']} día al futuro. Con origen T solo se usan valores fechados en "
        f"T−{f['rezago_minimo_desde_el_origen']} o antes."
    )
    st.altair_chart(graficas.pesos_fuga(r), width="stretch")
    if eda is not None:
        w = eda["sistema_actual"]["wape_pct"]
        c = st.columns(3)
        c[0].metric(
            "WAPE del sistema con la señal de T (con fuga)", pct(w["senal_T_factor_ajustado"]), border=True
        )
        c[1].metric("Con la señal de T−1 (sin fuga)", pct(w["senal_T_menos_1_factor_ajustado"]), border=True)
        c[2].metric("Media móvil de 7 días", pct(w["ma7"]), border=True)
        st.caption(
            "Suma de T+1 a T+7 con orígenes diarios (protocolo del EDA). La referencia válida para cualquier "
            "modelo es la de T−1."
        )


def calendario(eda: dict[str, Any]) -> None:
    st.subheader("Efectos del calendario")
    demanda = eda["demanda"]
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.altair_chart(graficas.efectos_calendario(eda), width="stretch")
        st.caption(
            "Contra la misma serie, mes y día de la semana en días limpios. La barra es el rango entre "
            "categorías: el calendario es multiplicativo y casi igual en todas."
        )
    with derecha:
        st.altair_chart(graficas.perfil_semanal(eda), width="stretch")
        t = demanda["tendencia"]
        st.caption(
            f"Crecimiento interanual: {pct(t['crecimiento_interanual_pct'], 1, signo=True)}; el mes más "
            f"alto es {t['mes_mas_alto']['mes']} (índice {t['mes_mas_alto']['indice']})."
        )
    st.info(BUEN_FIN + " Lo mismo pasa con Semana Santa.", icon=":material/event_busy:")
    with st.expander("Festivos y días atípicos"):
        st.dataframe(pl.DataFrame(demanda["festivos"]), hide_index=True, width="stretch")
        atipicos = demanda["dias_atipicos"]
        st.caption(
            f"Días atípicos (total por debajo de {atipicos['umbral']} × la mediana): "
            + (", ".join(atipicos["fechas"]) or "ninguno")
            + "."
        )
    with st.expander("Implicaciones para modelar"):
        for implicacion in eda["implicaciones"]:
            st.markdown(f"- {implicacion}")


reporte = datos.artefacto("calidad/reporte.json")
eda = datos.artefacto("eda/resumen.json")
if reporte is not None:
    contratos_y_malla(reporte)
    nulos_y_conteos(reporte)
    fuga(reporte, eda)
    with st.expander("Decisiones de calidad"):
        for decision in reporte["decisiones"]:
            st.markdown(f"- {decision}")
if eda is not None:
    calendario(eda)
