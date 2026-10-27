"""Pronóstico: la serie de una tienda × categoría en el backtest, con el real, el campeón y el retador.

Es la única página con filas de datos, y solo las de la serie que se elige. El catálogo de tiendas sale de
las columnas de tienda del panel; si el panel no existe, se elige la tienda solo por su identificador.
"""

from typing import Any

import polars as pl
import streamlit as st

from demanda.app import datos, graficas
from demanda.app.paginas import BUEN_FIN, si_no
from demanda.seleccion.campeon import campeon_y_retador

FILTROS = {"store_format": "Formato", "region": "Región", "socioeconomic_level": "NSE"}

st.title("Pronóstico")
st.caption("Elige una tienda, una categoría y un corte del backtest: el origen T y los días de T+1 a T+h.")


def elegir_tienda(ids: list[str], catalogo: pl.DataFrame | None) -> tuple[str | None, dict[str, Any] | None]:
    """La tienda elegida y su ficha. Con el catálogo, los filtros de formato, región y NSE se encadenan."""
    if catalogo is None:
        return st.selectbox("Tienda", ids), None
    tiendas = catalogo.filter(pl.col("store_id").is_in(ids))
    for columna, (campo, etiqueta) in zip(st.columns(len(FILTROS)), FILTROS.items(), strict=True):
        opciones = sorted(tiendas[campo].unique().to_list())
        elegidos = columna.multiselect(etiqueta, opciones, placeholder="todos", key=f"filtro_{campo}")
        if elegidos:
            tiendas = tiendas.filter(pl.col(campo).is_in(elegidos))
    if tiendas.is_empty():
        st.info("Ninguna tienda cumple los filtros.")
        return None, None
    fichas = {f["store_id"]: f for f in tiendas.iter_rows(named=True)}
    tienda = st.selectbox(
        "Tienda",
        list(fichas),
        format_func=lambda s: (
            f"{s} · {fichas[s]['store_format']} · {fichas[s]['region']} · NSE "
            f"{fichas[s]['socioeconomic_level']}"
        ),
    )
    return tienda, None if tienda is None else fichas[tienda]


def ficha(f: dict[str, Any]) -> None:
    st.caption(
        f"**{f['store_id']}** · formato {f['store_format']} · región {f['region']} · NSE "
        f"{f['socioeconomic_level']} · {f['size_sqm']:,} m² · {f['num_checkouts']} cajas · abrió en "
        f"{f['opening_year']} · farmacia: {si_no(f['has_pharmacy'])} · gasolinera: "
        f"{si_no(f['has_fuel_station'])}"
    )


def mostrar(
    catalogo: dict[str, list[str]],
    tiendas: pl.DataFrame | None,
    seleccion: dict[str, Any],
    metricas: dict[str, Any],
) -> None:
    campeon, retador = campeon_y_retador(seleccion)
    tienda, f = elegir_tienda(catalogo["tiendas"], tiendas)
    if tienda is None:
        return
    if f is not None:
        ficha(f)
    eventos = metricas["protocolo"]["evento_por_corte"]
    origenes = catalogo["origenes"]
    izquierda, centro, derecha = st.columns([2, 2, 5])
    categoria = izquierda.selectbox("Categoría", catalogo["categorias"])
    todos = (
        centro.radio(
            "Cortes",
            ["uno", "todos"],
            format_func=lambda v: {"uno": "uno", "todos": f"los {len(origenes)} seguidos"}[v],
            horizontal=True,
        )
        == "todos"
    )
    if todos:
        elegidos = origenes
    else:
        corte = derecha.select_slider(
            "Corte (origen T)",
            origenes,
            value=origenes[-1],
            format_func=lambda o: f"{o} · {eventos.get(o, '—')}",
        )
        elegidos = [corte]
    serie = datos.serie(tienda, categoria, (campeon, retador))
    if serie is None or categoria is None:
        return
    serie = serie.filter(pl.col("origen").cast(pl.String).is_in(elegidos))
    st.altair_chart(graficas.pronostico(serie, campeon, retador), width="stretch")

    puntuales = [m for m in (campeon, retador) if serie.filter(pl.col("modelo") == m)["p10"].is_null().all()]
    st.caption(
        "Campeón y retador de `seleccion.json`. La banda es el P10–P90 de cada modelo que lo tiene"
        + (f"; {', '.join(puntuales)} es puntual y no tiene intervalo." if puntuales else ".")
        + " Los días sin `units_sold` no tienen real."
    )
    if any(eventos.get(o) in ("buen_fin", "post_buen_fin") for o in elegidos):
        st.info(BUEN_FIN, icon=":material/event_busy:")
    with st.expander("La serie en tabla"):
        st.dataframe(
            serie.select(
                "store_id", "category", "modelo", "origen", "date", "h", "y", "yhat", "p10", "p90"
            ).sort("modelo", "date"),
            hide_index=True,
            width="stretch",
        )


tiendas = datos.tiendas()
fuentes = datos.cargar("modelos/seleccion.json", "backtest/metricas.json")
catalogo = datos.catalogo_pronosticos()
if fuentes is not None and catalogo is not None:
    seleccion, metricas = fuentes
    mostrar(catalogo, tiendas, seleccion, metricas)
