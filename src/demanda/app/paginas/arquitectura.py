"""Arquitectura: el diagrama de la carpeta más reciente de `.archify/` y el flujo `make` con su estado."""

import polars as pl
import streamlit as st

from demanda.app import datos

st.title("Arquitectura")

diagrama = datos.diagrama()
if diagrama is not None:
    ruta, html = diagrama
    st.caption(f"`{ruta.parent.name}/{ruta.name}`: la carpeta más reciente de `.archify/`.")
    st.iframe(html, height=900, alt="Diagrama de arquitectura de la solución")

st.subheader("Flujo `make`")
st.caption(
    "Cada paso lee lo que escribieron los anteriores; el estado dice si sus salidas ya existen. "
    "`make ajuste` (la rejilla de LightGBM) es opcional: sus hiperparámetros ya están fijos en el código."
)
st.dataframe(
    pl.DataFrame(datos.estado_flujo(datos.config())), hide_index=True, width="stretch", height="content"
)
