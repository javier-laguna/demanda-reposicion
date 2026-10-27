"""Promociones (E6): el experimento natural con `has_promotion`. Balance, efecto principal y por canal,
heterogeneidad, dinámica, potencia y si la promoción se paga."""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos, graficas
from demanda.app.paginas import insights, pct, si_no, valor_p

st.title("Promociones")
insights("causal")


def ic(e: dict[str, Any]) -> str:
    """Un efecto con su IC95 y su valor p, en una línea."""
    bajo, alto = (pct(x, signo=True) for x in e["ic95_pct"])
    return f"{pct(e['efecto_pct'], signo=True)} (IC95 de {bajo} a {alto}; p = {valor_p(e['p_valor'])})"


def balance(r: dict[str, Any]) -> None:
    st.subheader("Balance de la asignación")
    m, b = r["muestra"], r["balance"]
    c = st.columns(4)
    c[0].metric(
        "Filas analizadas",
        f"{m['filas']:,}",
        help=f"{m['series']} series de {m['tiendas']} tiendas, del {m['desde']} al {m['hasta']}.",
        border=True,
    )
    c[1].metric("Días con promoción", pct(m["tasa_promocion_pct"]), border=True)
    c[2].metric(
        "B1 · segmento y calendario",
        f"p = {valor_p(b['b1']['p_valor'])}",
        help=b["b1"]["descripcion"],
        border=True,
    )
    c[3].metric(
        "B3 · promoción futura",
        f"p = {valor_p(b['b3']['p_valor'])}",
        help=b["b3"]["descripcion"],
        border=True,
    )
    medicion = [p for p in b["fallan"] if p in b["pruebas_de_medicion"]]
    if b["asignacion_se_sostiene"]:
        st.success(
            "Las pruebas de la asignación pasan: la promoción se asignó al azar."
            + (
                f" Falla {', '.join(medicion)}, que mide el registro del mismo día y no la asignación."
                if medicion
                else ""
            ),
            icon=":material/check_circle:",
        )
    else:
        st.error(f"La asignación no se sostiene: fallan {', '.join(b['fallan'])}.", icon=":material/error:")
    familia = b["familia"]
    with st.expander(f"Las {familia['n_pruebas']} pruebas de balance, con Holm"):
        st.dataframe(
            pl.DataFrame(
                [{"prueba": p["prueba"], "p de Holm": valor_p(p["p_holm"])} for p in familia["pruebas"]]
            ),
            hide_index=True,
            width="stretch",
        )


def efecto(r: dict[str, Any]) -> None:
    st.subheader("Efecto principal y por canal")
    e, aa = r["efecto_principal"], r["calibracion_aa"]
    bajo, alto = e["ic95_pct"]
    c = st.columns(3)
    c[0].metric("Efecto en unidades", pct(e["efecto_pct"], signo=True), help=e["modelo"], border=True)
    c[1].metric("IC95", f"{pct(bajo, signo=True)} a {pct(alto, signo=True)}", border=True)
    c[2].metric(
        "p agrupada por tienda · por permutaciones",
        f"{valor_p(e['p_valor'])} · {valor_p(e['p_aleatorizacion'])}",
        border=True,
    )
    st.caption(
        f"Significativo al 5 %: {si_no(e['significativo'])}. Calibración A/A: {pct(aa['tasa_pct'], 1)} de "
        "rechazos "
        f"con placebos (intervalo del 99 % de {pct(aa['intervalo_99_pct'][0], 1)} a "
        f"{pct(aa['intervalo_99_pct'][1], 1)}; dentro: {si_no(aa['dentro'])})."
    )
    st.altair_chart(graficas.estimaciones(r["canales"]), width="stretch")
    for nombre, canal in r["canales"].items():
        nota = f" {canal['advertencia']}" if canal.get("advertencia") else ""
        st.caption(f"`{nombre}`: {canal['resultado']}.{nota}")
    with st.expander("Robustez"):
        st.altair_chart(graficas.estimaciones(r["robustez"]), width="stretch")
        for nombre, v in r["robustez"].items():
            st.caption(
                f"`{nombre}`: {v['descripcion']}"
                + ("" if "efecto_pct" in v else f" (p = {valor_p(v['p_valor'])}).")
            )


def heterogeneidad(r: dict[str, Any]) -> None:
    st.subheader("Heterogeneidad y lift de equilibrio")
    h, negocio = r["heterogeneidad"], r["negocio"]
    descuentos = {d["descuento"]: d["origen"] for d in negocio["por_descuento"]}
    opciones = list(descuentos)
    implicito = next((i for i, d in enumerate(opciones) if descuentos[d] == "implicito"), 0)
    descuento = st.selectbox(
        "Descuento (supuesto)",
        opciones,
        index=implicito,
        format_func=lambda d: (
            f"{100 * d:.3g} % · {'implícito en el precio' if descuentos[d] == 'implicito' else 'rejilla'}"
        ),
    )
    if descuento is None:
        return
    st.altair_chart(graficas.efecto_por_categoria(r, descuento), width="stretch")
    igualdad = h["categoria"]["igualdad"]["p_valor"]
    significativos = ", ".join(h["familia_holm"]["significativos"]) or "ninguno"
    st.caption(
        "El lift de equilibrio es el aumento de unidades con el que la promoción apenas se paga con el "
        f"margen y el descuento supuestos. Igualdad entre categorías: p = {valor_p(igualdad)}. Distintos "
        f"de cero con Holm: {significativos}."
    )
    for dimension, d in h.items():
        if dimension in ("categoria", "familia_holm"):
            continue
        st.markdown(f"**Por {dimension}** (igualdad: p = {valor_p(d['igualdad']['p_valor'])})")
        st.altair_chart(graficas.estimaciones(d["grupos"]), width="stretch")


def dinamica(r: dict[str, Any]) -> None:
    st.subheader("Dinámica y efecto cruzado")
    d, cruzado = r["dinamica"], r["cruzado"]
    st.altair_chart(graficas.estimaciones(d["coeficientes"]), width="stretch")
    st.caption(
        f"Coeficiente de la promoción del día t y de los {len(d['coeficientes']) - 1} días previos. Suma "
        f"de rezagos: {ic(d['suma_rezagos'])}; neto: {ic(d['neto'])}. Adelanto de compras: "
        f"{si_no(d['adelanto'])}."
    )
    st.markdown(
        f"Efecto sobre las otras categorías de la tienda: {ic(cruzado['otras_categorias'])}, "
        f"`{cruzado['interpretacion']}`."
    )


def potencia(r: dict[str, Any]) -> None:
    st.subheader("Potencia")
    p = r["potencia"]
    semanal = p["diseno_semanal"]
    izquierda, derecha = st.columns([1, 2])
    izquierda.metric(
        "Efecto mínimo detectable hoy",
        pct(p["diseno_actual"]["global"]["mde_pct"]),
        help=f"α = {p['diseno_actual']['alfa']} y potencia {p['diseno_actual']['potencia']}, por día.",
        border=True,
    )
    izquierda.caption(
        "Semanas de un diseño semanal para detectar: "
        + ", ".join(f"{f['efecto_pct']:g} %: {f['semanas'] or '—'}" for f in semanal["semanas_necesarias"])
        + "."
    )
    with derecha:
        st.altair_chart(graficas.potencia_semanal(r), width="stretch")
        st.caption(semanal["descripcion"])


def negocio(r: dict[str, Any]) -> None:
    st.subheader("¿Se paga la promoción?")
    n = r["negocio"]
    st.info(
        "Los descuentos y los márgenes son supuestos (los márgenes son los mismos del newsvendor); el "
        "descuento implícito se estima con el efecto sobre el precio por unidad.",
        icon=":material/info:",
    )
    st.dataframe(
        pl.DataFrame(
            [
                {
                    "descuento (supuesto)": f"{100 * f['descuento']:.3g} %",
                    "origen": f["origen"],
                    "se paga": ", ".join(f["se_paga"]) or "—",
                    "no se paga": ", ".join(f["no_se_paga"]) or "—",
                    "indeterminado": ", ".join(f["indeterminado"]) or "—",
                }
                for f in n["por_descuento"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    implicito = n["descuento_implicito"]
    st.caption(f"Descuento implícito: {implicito['regla']}; aplica: {si_no(implicito['aplica'])}.")
    st.markdown(r["conclusion"]["texto"])


promociones = datos.artefacto("causal/promociones.json")
if promociones is not None:
    balance(promociones)
    efecto(promociones)
    heterogeneidad(promociones)
    dinamica(promociones)
    potencia(promociones)
    negocio(promociones)
