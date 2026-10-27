"""Punto de entrada del tablero: `uv run demanda app` (o `make app`) lo corre con `streamlit run`.

Registra las siete páginas con `st.navigation`. El tema no se fija: sigue la preferencia del navegador
(claro u oscuro, y se cambia en el menú de Streamlit), y las gráficas toman la paleta de ese tema.
"""

import streamlit as st

st.set_page_config(page_title="Pronóstico de demanda", page_icon=":material/insights:", layout="wide")

# Las rutas son relativas a este archivo
PAGINAS = [
    st.Page("paginas/resumen.py", title="Resumen", icon=":material/dashboard:", default=True),
    st.Page("paginas/datos.py", title="Datos", icon=":material/dataset:"),
    st.Page("paginas/modelos.py", title="Modelos", icon=":material/model_training:"),
    st.Page("paginas/pronostico.py", title="Pronóstico", icon=":material/show_chart:"),
    st.Page("paginas/decision.py", title="Decisión y drift", icon=":material/inventory_2:"),
    st.Page("paginas/promociones.py", title="Promociones", icon=":material/sell:"),
    st.Page("paginas/arquitectura.py", title="Arquitectura", icon=":material/account_tree:"),
]

pagina = st.navigation(PAGINAS)
st.sidebar.caption(
    "El tablero solo lee artefactos y no recalcula nada. Si cambian los datos, corre el pipeline "
    "(página Arquitectura) y recarga."
)
pagina.run()
