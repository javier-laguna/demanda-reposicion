"""Las gráficas del tablero: funciones puras de un artefacto (dict) o un DataFrame a una gráfica de Altair.

- No dependen de Streamlit, así que se prueban solas.
- Salvo el pronóstico, no fijan colores: `st.altair_chart` aplica el tema de Streamlit, con su paleta y sus
  colores de texto, y así se leen igual en tema claro y oscuro. Lo que distingue series es el dominio de cada
  escala, fijo para que una serie tenga el mismo color en todas las vistas.
- Los intervalos van como bandas (`mark_area`) o barras de error (`mark_errorbar`), nunca solo como puntos.
- Solo cambian unidades para mostrar (pesos a millones); las cifras salen tal cual de los artefactos.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import altair as alt
import polars as pl

ALTO = 300
# Streamlit ajusta la gráfica al alto dado, con ejes y leyenda adentro: las de barras suman este margen
MARGEN = 90
TITULOS_METRICA = {
    "wape_pct": "WAPE (%)",
    "sesgo_pct": "sesgo (%) · positivo = sobrepronóstico",
    "cobertura_80_pct": "cobertura del P10–P90 (%)",
}
EFECTOS_CALENDARIO = {
    "buen_fin": "Buen Fin",
    "quincena": "quincena",
    "fin_de_semana": "fin de semana",
    "navidad": "temporada navideña",
    "semana_santa": "Semana Santa",
}
DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
CLASIFICACIONES = ("inesperada", "esperada")
CAPAS = {"calidad": "calidad de los datos", "entradas": "entradas del modelo", "desempeno": "desempeño"}
# El pronóstico fija sus colores para que el real no se confunda con un modelo: gris neutro para el real, azul
# para el campeón y naranja para el retador, con tonos medios que se leen sobre fondo claro y oscuro
COLORES_PRONOSTICO = {"real": "#8C8C8C", "campeon": "#3B82F6", "retador": "#F58518"}


def _capas(*graficas: alt.Chart, **propiedades: Any) -> alt.LayerChart:
    """Superpone gráficas: `alt.layer` sin el caso de facetas, que aquí no se usa; así el tipo es exacto."""
    return alt.LayerChart(layer=list(graficas), **propiedades)


def _leyenda(columnas: int | None = None, valores: Sequence[str] | None = None) -> alt.Legend:
    """La leyenda arriba, sin título; con `columnas`, en rejilla (para las gráficas a media página), y con
    `valores`, solo esos elementos del dominio."""
    # Símbolos opacos: si una capa es una banda, la leyenda no hereda su transparencia
    opciones: dict[str, Any] = {"orient": "top", "title": None, "labelLimit": 320, "symbolOpacity": 1}
    if columnas is not None:
        opciones["columns"] = columnas
    if valores is not None:
        opciones["values"] = list(valores)
    return alt.Legend(**opciones)


def _categorias(campo: str, orden: Sequence[str] | str) -> alt.Y:
    """El eje de categorías de las gráficas horizontales, sin recortar etiquetas largas."""
    return alt.Y(
        campo,
        sort=orden if isinstance(orden, str) else list(orden),
        title=None,
        axis=alt.Axis(labelLimit=260),
    )


def _regla(x: float) -> alt.Chart:
    """Una línea vertical de referencia en `x` (el cero de un efecto, el 1 de un multiplicador)."""
    return alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")


def _con_ic(
    df: pl.DataFrame, eje: str, titulo: str, orden: Sequence[str], color: alt.Color | None = None
) -> list[alt.Chart]:
    """Punto en `efecto` y barra de error de `bajo` a `alto` por cada valor de `eje`."""
    y = _categorias(f"{eje}:N", orden)
    tooltip = [
        alt.Tooltip(f"{eje}:N", title=""),
        alt.Tooltip("efecto:Q", title="efecto (%)", format="+.3f"),
        alt.Tooltip("bajo:Q", title="IC95 desde", format="+.3f"),
        alt.Tooltip("alto:Q", title="IC95 hasta", format="+.3f"),
        alt.Tooltip("p:Q", title="p", format=".3g"),
    ]
    barras = alt.Chart(df).mark_errorbar(ticks=True).encode(y=y, x=alt.X("bajo:Q", title=titulo), x2="alto:Q")
    puntos = alt.Chart(df).mark_point(filled=True, size=70).encode(y=y, x="efecto:Q", tooltip=tooltip)
    if color is not None:
        barras, puntos = barras.encode(color=color), puntos.encode(color=color)
    return [barras, puntos]


def _estimacion(nombre: str, e: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "nombre": nombre,
        "efecto": e["efecto_pct"],
        "bajo": e["ic95_pct"][0],
        "alto": e["ic95_pct"][1],
        "p": e["p_valor"],
        **extra,
    }


ESQUEMA_IC = {
    "nombre": pl.String,
    "efecto": pl.Float64,
    "bajo": pl.Float64,
    "alto": pl.Float64,
    "p": pl.Float64,
}


# --- datos ---------------------------------------------------------------------------------------------


def nulos(reporte: Mapping[str, Any]) -> alt.Chart:
    """El % de nulos por columna del reporte de calidad, coloreado por su patrón (disperso o en bloque)."""
    df = pl.DataFrame(
        [
            {"columna": c, "pct": n["pct"], "nulos": n["nulos"], "patron": n["patron"]}
            for c, n in reporte["nulos"].items()
        ],
        schema={"columna": pl.String, "pct": pl.Float64, "nulos": pl.Int64, "patron": pl.String},
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=_categorias("columna:N", "-x"),
            x=alt.X("pct:Q", title="filas con nulo (%)"),
            color=alt.Color("patron:N", scale=alt.Scale(domain=["disperso", "bloque"]), legend=_leyenda()),
            tooltip=[
                alt.Tooltip("columna:N"),
                alt.Tooltip("nulos:Q", format=","),
                alt.Tooltip("pct:Q", title="%", format=".2f"),
                alt.Tooltip("patron:N", title="patrón"),
            ],
        )
        .properties(height=32 * df.height + MARGEN)
    )


def pesos_fuga(reporte: Mapping[str, Any]) -> alt.Chart:
    """El peso de cada desfase de `units_sold` en `replenishment_signal`: la ventana que mejor la explica y,
    dentro de ella, los días que la señal ya conoce al fecharse (el mismo día y después: la fuga)."""
    fuga = reporte["fuga_replenishment_signal"]
    ventana = fuga["mejor_ventana"]
    zonas = ("fuera de la ventana", "ventana: días previos", "ventana: el día y después (fuga)")

    def zona(desfase: int) -> str:
        if not ventana["desde"] <= desfase <= ventana["hasta"]:
            return zonas[0]
        return zonas[1] if desfase < 0 else zonas[2]

    filas = sorted(
        ({"desfase": int(k), "peso": peso} for k, peso in fuga["pesos_por_desfase"].items()),
        key=lambda f: f["desfase"],
    )
    df = pl.DataFrame(
        [{**f, "etiqueta": f"t{f['desfase']:+d}", "zona": zona(f["desfase"])} for f in filas],
        schema={"desfase": pl.Int64, "peso": pl.Float64, "etiqueta": pl.String, "zona": pl.String},
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "etiqueta:N",
                sort=df["etiqueta"].to_list(),
                title="ventas del día t + desfase",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("peso:Q", title="peso en la señal fechada en t"),
            color=alt.Color("zona:N", scale=alt.Scale(domain=list(zonas)), legend=_leyenda()),
            tooltip=[alt.Tooltip("etiqueta:N", title="desfase"), alt.Tooltip("peso:Q", format=".4f")],
        )
        .properties(height=ALTO)
    )


def efectos_calendario(resumen: Mapping[str, Any]) -> alt.LayerChart:
    """El multiplicador de cada efecto del calendario contra días limpios, con el rango entre categorías
    como barra de error y la línea en 1 (sin efecto). Los efectos que el EDA no pudo medir no aparecen."""
    demanda = resumen["demanda"]
    rangos = demanda.get("dispersion_efectos_por_categoria", {})
    dias = demanda.get("dias_por_efecto", {})
    df = pl.DataFrame(
        [
            {
                "efecto": EFECTOS_CALENDARIO.get(k, k),
                "multiplicador": m,
                "min": rangos.get(k, {}).get("min"),
                "max": rangos.get(k, {}).get("max"),
                "dias": dias.get(k),
            }
            for k, m in demanda["efectos_calendario"].items()
            if m is not None
        ],
        schema={
            "efecto": pl.String,
            "multiplicador": pl.Float64,
            "min": pl.Float64,
            "max": pl.Float64,
            "dias": pl.Int64,
        },
    )
    y = _categorias("efecto:N", df.sort("multiplicador", descending=True)["efecto"].to_list())
    rango = (
        alt.Chart(df)
        .mark_errorbar(ticks=True)
        .encode(
            y=y, x=alt.X("min:Q", title="multiplicador de la demanda (rango entre categorías)"), x2="max:Q"
        )
    )
    puntos = (
        alt.Chart(df)
        .mark_point(filled=True, size=80)
        .encode(
            y=y,
            x=alt.X("multiplicador:Q", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("efecto:N"),
                alt.Tooltip("multiplicador:Q", format=".3f"),
                alt.Tooltip("min:Q", title="mínimo entre categorías", format=".3f"),
                alt.Tooltip("max:Q", title="máximo entre categorías", format=".3f"),
                alt.Tooltip("dias:Q", title="días en los datos"),
            ],
        )
    )
    return _capas(rango, puntos, _regla(1.0), height=40 * df.height + MARGEN)


def perfil_semanal(resumen: Mapping[str, Any]) -> alt.LayerChart:
    """El índice de demanda por día de la semana (1 = el promedio), con la línea en 1."""
    perfil = resumen["demanda"]["estacionalidad_semanal"]["perfil_dia_semana"]
    df = pl.DataFrame(
        [{"dia": d, "indice": perfil[d]} for d in DIAS if d in perfil],
        schema={"dia": pl.String, "indice": pl.Float64},
    )
    barras = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("dia:N", sort=list(DIAS), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("indice:Q", title="índice (1 = promedio)"),
            tooltip=[alt.Tooltip("dia:N", title="día"), alt.Tooltip("indice:Q", format=".3f")],
        )
    )
    uno = alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    return _capas(barras, uno, height=ALTO)


# --- modelos y pronóstico ------------------------------------------------------------------------------


def _segmentos(
    metricas: Mapping[str, Any], segmentos: Sequence[str], metrica: str, granularidad: str
) -> pl.DataFrame:
    filas = [
        {k: f[k] for k in ("segmento", "nivel", "modelo", "valor", "n", "n_total")}
        for f in metricas["segmentos"]
        if f["segmento"] in segmentos and f["metrica"] == metrica and f["granularidad"] == granularidad
    ]
    esquema = {
        "segmento": pl.String,
        "nivel": pl.String,
        "modelo": pl.String,
        "valor": pl.Float64,
        "n": pl.Int64,
        "n_total": pl.Int64,
    }
    return pl.DataFrame(filas, schema=esquema)


def _barras_por_modelo(
    df: pl.DataFrame, modelos: Sequence[str], titulo: str, formato: str = ".2f"
) -> alt.Chart:
    """Barras horizontales agrupadas: un grupo por nivel y una barra por modelo. El dominio de color es el de
    todos los `modelos`, para que cada uno tenga su color en todas las gráficas; la leyenda solo lista los que
    tienen datos."""
    niveles = df["nivel"].unique(maintain_order=True).to_list()
    presentes = [m for m in modelos if m in set(df["modelo"].to_list())]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=_categorias("nivel:N", niveles),
            x=alt.X("valor:Q", title=titulo),
            yOffset=alt.YOffset("modelo:N", sort=presentes),
            color=alt.Color(
                "modelo:N",
                scale=alt.Scale(domain=list(modelos)),
                legend=_leyenda(columnas=3, valores=presentes),
            ),
            tooltip=[
                alt.Tooltip("nivel:N"),
                alt.Tooltip("modelo:N"),
                alt.Tooltip("valor:Q", title=titulo, format=formato),
                alt.Tooltip("n:Q", title="puntos evaluados", format=","),
                alt.Tooltip("n_total:Q", title="puntos del segmento", format=","),
            ],
        )
        .properties(height=14 * len(niveles) * max(len(presentes), 1) + MARGEN)
    )


def metrica_por_segmento(
    metricas: Mapping[str, Any], segmento: str, metrica: str = "wape_pct", granularidad: str = "semanal"
) -> alt.LayerChart:
    """Una métrica de `metricas.json` (WAPE o sesgo) por nivel del segmento y modelo, con la línea en cero."""
    df = _segmentos(metricas, [segmento], metrica, granularidad)
    barras = _barras_por_modelo(df, metricas["protocolo"]["modelos"], TITULOS_METRICA[metrica])
    return _capas(barras, _regla(0.0))


def cobertura_intervalos(metricas: Mapping[str, Any], segmento: str = "evento") -> alt.LayerChart:
    """La cobertura diaria del P10–P90 de los modelos con intervalo, en el global y por nivel del segmento,
    con la línea del 80 % nominal."""
    df = _segmentos(metricas, ["global", segmento], "cobertura_80_pct", "diaria")
    barras = _barras_por_modelo(
        df, metricas["protocolo"]["modelos"], TITULOS_METRICA["cobertura_80_pct"], ".1f"
    )
    eje = alt.X("valor:Q", title=TITULOS_METRICA["cobertura_80_pct"], scale=alt.Scale(domain=[0, 100]))
    return _capas(barras.encode(x=eje), _regla(80.0))


def pronostico(serie: pl.DataFrame, campeon: str, retador: str) -> alt.LayerChart:
    """El real y el pronóstico diario del campeón y del retador, con la banda P10–P90 de cada modelo que la
    tiene. `serie` tiene las columnas de `pronosticos.parquet` (una tienda × categoría, uno o varios cortes);
    cada corte es un tramo propio de la línea y de la banda."""
    nombres = {campeon: f"campeón · {campeon}", retador: f"retador · {retador}"}
    dominio = ["real", *nombres.values()]
    columnas = pl.Schema(
        {
            "origen": pl.String,
            "date": pl.Date,
            "serie": pl.String,
            "valor": pl.Float64,
            "p10": pl.Float64,
            "p90": pl.Float64,
        }
    )
    modelos = serie.filter(pl.col("modelo").is_in(list(nombres)))
    pronosticos = modelos.select(
        pl.col("origen").cast(pl.String),
        "date",
        pl.col("modelo").replace_strict(nombres, return_dtype=pl.String).alias("serie"),
        pl.col("yhat").alias("valor"),
        "p10",
        "p90",
    ).cast(columnas)
    # El real es uno por día: no se corta por corte
    real = (
        serie.select("date", pl.col("y").alias("valor"))
        .unique("date")
        .sort("date")
        .select(
            pl.lit("real").alias("origen"),
            "date",
            pl.lit("real").alias("serie"),
            "valor",
            pl.lit(None, pl.Float64).alias("p10"),
            pl.lit(None, pl.Float64).alias("p90"),
        )
        .cast(columnas)
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d/%m", tickCount="day"))
    # Cada banda va en el color de su modelo, con transparencia
    escala = alt.Scale(domain=dominio, range=list(COLORES_PRONOSTICO.values()))
    color = alt.Color("serie:N", scale=escala, legend=_leyenda())
    tooltip = [
        alt.Tooltip("date:T", title="fecha", format="%Y-%m-%d"),
        alt.Tooltip("serie:N"),
        alt.Tooltip("valor:Q", title="unidades", format=",.0f"),
        alt.Tooltip("p10:Q", title="P10", format=",.0f"),
        alt.Tooltip("p90:Q", title="P90", format=",.0f"),
    ]
    banda = (
        alt.Chart(pronosticos.filter(pl.col("p10").is_not_null()))
        .mark_area(opacity=0.2)
        .encode(
            x=x,
            y=alt.Y("p10:Q", title="unidades"),
            y2="p90:Q",
            color=color,
            detail="origen:N",
        )
    )
    lineas = (
        alt.Chart(pl.concat([real, pronosticos]))
        .mark_line(point=True)
        .encode(
            x=x,
            y=alt.Y("valor:Q", title="unidades"),
            color=color,
            strokeWidth=alt.condition(alt.datum.serie == "real", alt.value(3), alt.value(2)),
            detail="origen:N",
            tooltip=tooltip,
        )
    )
    return _capas(banda, lineas, height=ALTO + 60)


# --- decisión y drift ----------------------------------------------------------------------------------


def costo_por_politica(impacto: Mapping[str, Any]) -> alt.Chart:
    """El costo de cada política en el global y por evento, en millones de pesos (con costos supuestos)."""
    politicas = list(impacto["politicas"])
    df = pl.DataFrame(
        [
            {
                "nivel": f["nivel"],
                "politica": f["politica"],
                "costo": f["costo"] / 1e6,
                "quiebre_pct": f["quiebre_pct"],
                "sobreinventario_pct": f["sobreinventario_pct"],
            }
            for f in impacto["impacto"]
            if f["segmento"] in ("global", "evento")
        ],
        schema={
            "nivel": pl.String,
            "politica": pl.String,
            "costo": pl.Float64,
            "quiebre_pct": pl.Float64,
            "sobreinventario_pct": pl.Float64,
        },
    )
    niveles = df["nivel"].unique(maintain_order=True).to_list()
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=_categorias("nivel:N", niveles),
            x=alt.X("costo:Q", title="costo (millones de pesos, costos supuestos)"),
            yOffset=alt.YOffset("politica:N", sort=politicas),
            color=alt.Color("politica:N", scale=alt.Scale(domain=politicas), legend=_leyenda()),
            tooltip=[
                alt.Tooltip("nivel:N"),
                alt.Tooltip("politica:N", title="política"),
                alt.Tooltip("costo:Q", title="costo (M$)", format=",.1f"),
                alt.Tooltip("quiebre_pct:Q", title="quiebre (%)", format=".2f"),
                alt.Tooltip("sobreinventario_pct:Q", title="sobreinventario (%)", format=".2f"),
            ],
        )
        .properties(height=16 * len(niveles) * len(politicas) + MARGEN)
    )


def sensibilidad_costo(impacto: Mapping[str, Any]) -> alt.Chart:
    """El costo global de cada política con la misma fracción crítica en todas las categorías."""
    politicas = list(impacto["politicas"])
    df = pl.DataFrame(
        [
            {"fraccion_critica": f["fraccion_critica"], "politica": p, "costo": v["costo"] / 1e6}
            for f in impacto["sensibilidad"]["fracciones"]
            for p, v in f["global"].items()
        ],
        schema={"fraccion_critica": pl.Float64, "politica": pl.String, "costo": pl.Float64},
    )
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "fraccion_critica:Q",
                title="fracción crítica (igual en todas las categorías)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("costo:Q", title="costo global (millones de pesos, supuestos)"),
            color=alt.Color("politica:N", scale=alt.Scale(domain=politicas), legend=_leyenda()),
            tooltip=[
                alt.Tooltip("fraccion_critica:Q", title="fracción crítica"),
                alt.Tooltip("politica:N", title="política"),
                alt.Tooltip("costo:Q", title="costo (M$)", format=",.1f"),
            ],
        )
        .properties(height=ALTO)
    )


def alertas_por_semana(monitor: Mapping[str, Any], capa: str) -> alt.LayerChart:
    """Las alertas de una capa del monitor por semana, apiladas en inesperadas y esperadas (por un día
    especial o un evento del backtest), con la marca de la primera semana del backtest. Las de severidad
    «vigilar» no cuentan."""
    semanas = monitor["semanas"]
    origenes = [s["origen"] for s in semanas]
    # Este conteo no está en `monitor.json`: `resumen.por_semana` agrega por semana sin separar la capa, y
    # `resumen.<periodo>.por_capa`, por capa sin separar la semana. Se cuenta desde las alertas de cada
    # semana; las pruebas exigen que la suma de las capas coincida con `por_semana`.
    filas = [
        {
            "origen": s["origen"],
            "periodo": s["periodo"],
            "evento": s["evento"],
            "clasificacion": clase,
            "alertas": sum(
                a["capa"] == capa and a["severidad"] == "alerta" and a["clasificacion"] == clase
                for a in s["alertas"]
            ),
        }
        for s in semanas
        for clase in CLASIFICACIONES
    ]
    df = pl.DataFrame(
        filas,
        schema={
            "origen": pl.String,
            "periodo": pl.String,
            "evento": pl.String,
            "clasificacion": pl.String,
            "alertas": pl.Int64,
        },
    )
    x = alt.X(
        "origen:O",
        scale=alt.Scale(domain=origenes),
        title=None,
        axis=alt.Axis(labelAngle=-90, labelOverlap=True),
    )
    barras = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=x,
            y=alt.Y("alertas:Q", title="alertas", axis=alt.Axis(tickMinStep=1, format="d")),
            color=alt.Color(
                "clasificacion:N", scale=alt.Scale(domain=list(CLASIFICACIONES)), legend=_leyenda()
            ),
            order=alt.Order("clasificacion:N", sort="descending"),
            tooltip=[
                alt.Tooltip("origen:O", title="semana (origen T)"),
                alt.Tooltip("periodo:N"),
                alt.Tooltip("evento:N"),
                alt.Tooltip("clasificacion:N", title="clasificación"),
                alt.Tooltip("alertas:Q"),
            ],
        )
    )
    capas = [barras]
    inicio = next((s["origen"] for s in semanas if s["periodo"] == "backtest"), None)
    if inicio is not None:
        marca = alt.Chart(pl.DataFrame({"origen": [inicio], "texto": ["backtest →"]}))
        capas += [
            marca.mark_rule(strokeDash=[4, 4]).encode(x="origen:O"),
            marca.mark_text(align="left", baseline="top", dx=4, dy=2).encode(
                x="origen:O", y=alt.value(0), text="texto:N"
            ),
        ]
    return _capas(*capas, height=280, title=CAPAS.get(capa, capa))


def vigilancia_retador(monitor: Mapping[str, Any]) -> alt.Chart:
    """El WAPE semanal acumulado del campeón y del retador al cerrar cada corte del backtest."""
    modelos = [monitor["campeon"], monitor["retador"]]
    df = pl.DataFrame(
        [
            {"corte": c["corte"], "modelo": m, "wape": v, "gana": c["cortes_gana_retador"]}
            for c in monitor["vigilancia_retador"]["por_corte"]
            for m, v in c["wape_acumulado_pct"].items()
        ],
        schema={"corte": pl.String, "modelo": pl.String, "wape": pl.Float64, "gana": pl.Int64},
    ).with_columns(pl.col("corte").str.to_date())
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("corte:T", title=None, axis=alt.Axis(format="%d/%m")),
            y=alt.Y("wape:Q", title="WAPE semanal acumulado (%)"),
            color=alt.Color("modelo:N", scale=alt.Scale(domain=modelos), legend=_leyenda()),
            tooltip=[
                alt.Tooltip("corte:T", format="%Y-%m-%d"),
                alt.Tooltip("modelo:N"),
                alt.Tooltip("wape:Q", title="WAPE acumulado (%)", format=".2f"),
                alt.Tooltip("gana:Q", title="cortes que gana el retador"),
            ],
        )
        .properties(height=ALTO)
    )


# --- promociones ---------------------------------------------------------------------------------------


def estimaciones(filas: Mapping[str, Mapping[str, Any]], titulo: str = "efecto (%)") -> alt.LayerChart:
    """Efectos con su IC95 (punto y barra de error) y la línea en cero. `filas` va de nombre a una estimación
    de `promociones.json`; las entradas sin efecto (una prueba de permutaciones, p. ej.) no aparecen."""
    df = pl.DataFrame([_estimacion(n, e) for n, e in filas.items() if "efecto_pct" in e], schema=ESQUEMA_IC)
    return _capas(
        *_con_ic(df, "nombre", titulo, df["nombre"].to_list()), _regla(0.0), height=36 * df.height + MARGEN
    )


def efecto_por_categoria(promociones: Mapping[str, Any], descuento: float) -> alt.LayerChart:
    """El efecto de la promoción en cada categoría con su IC95 y, para el descuento dado, el lift de
    equilibrio: el aumento de unidades con el que la promoción apenas se paga con el margen supuesto."""
    grupos = promociones["heterogeneidad"]["categoria"]["grupos"]
    marcas = ("efecto estimado (IC95)", "lift de equilibrio")
    efectos = pl.DataFrame(
        [_estimacion(c, g, marca=marcas[0]) for c, g in grupos.items()],
        schema={**ESQUEMA_IC, "marca": pl.String},
    )
    equilibrio = pl.DataFrame(
        [
            {"nombre": c, "lift": d["lift_equilibrio_pct"], "marca": marcas[1], "veredicto": d["veredicto"]}
            for c, n in promociones["negocio"]["por_categoria"].items()
            for d in n["por_descuento"]
            if math.isclose(d["descuento"], descuento)
        ],
        schema={"nombre": pl.String, "lift": pl.Float64, "marca": pl.String, "veredicto": pl.String},
    )
    color = alt.Color("marca:N", scale=alt.Scale(domain=list(marcas)), legend=_leyenda())
    orden = list(grupos)
    linea = (
        alt.Chart(equilibrio)
        .mark_tick(thickness=4, size=24)
        .encode(
            y=_categorias("nombre:N", orden),
            x="lift:Q",
            color=color,
            tooltip=[
                alt.Tooltip("nombre:N", title="categoría"),
                alt.Tooltip("lift:Q", title="lift de equilibrio (%)", format=".2f"),
                alt.Tooltip("veredicto:N"),
            ],
        )
    )
    capas = _con_ic(efectos, "nombre", "efecto sobre las unidades (%)", orden, color)
    return _capas(*capas, linea, _regla(0.0), height=40 * efectos.height + MARGEN)


def potencia_semanal(promociones: Mapping[str, Any]) -> alt.Chart:
    """El efecto mínimo detectable con un diseño de promoción por semana, según las semanas que dura."""
    diseno = promociones["potencia"]["diseno_semanal"]
    df = pl.DataFrame(
        [{"semanas": f["semanas"], "mde_pct": f["mde_pct"]} for f in diseno["por_semanas"]],
        schema={"semanas": pl.Int64, "mde_pct": pl.Float64},
    )
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("semanas:Q", title="semanas del diseño", axis=alt.Axis(format="d")),
            y=alt.Y("mde_pct:Q", title="efecto mínimo detectable (%)"),
            tooltip=[alt.Tooltip("semanas:Q"), alt.Tooltip("mde_pct:Q", title="MDE (%)", format=".2f")],
        )
        .properties(height=ALTO - 60)
    )
