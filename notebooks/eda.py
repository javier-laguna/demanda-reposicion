import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="EDA · demanda por tienda y categoría")


@app.cell
def _():
    import json
    from datetime import date
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from demanda.config import get_config
    from demanda.datos.contratos import LLAVE, SERIE
    from demanda.datos.fuente import sha256

    return LLAVE, Path, SERIE, alt, date, get_config, json, mo, pl, sha256


@app.cell
def _():
    def redondear(x, n=4):
        """Float redondeado; None si el valor es nulo, NaN o infinito (así el JSON nunca lleva NaN)."""
        if x is None:
            return None
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return round(x, n)

    def cifra(x, formato=".2f", sufijo=""):
        """Texto para el notebook: «s/d» cuando la cifra no existe en estos datos."""
        return "s/d" if x is None else f"{x:{formato}}{sufijo}"

    def tabla_md(encabezados, filas):
        lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
        lineas += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
        return "\n".join(lineas)

    return cifra, redondear, tabla_md


@app.cell
def _(date):
    UMBRAL_ATIPICO = 0.30  # un día es atípico si su demanda cae debajo de este múltiplo de la mediana
    INICIO_EVALUACION = date(2023, 7, 1)
    FACTOR_EXPLORACION = 0.53
    HORIZONTE = 7
    # Eventos de calendario que son banderas de fecha; el fin de semana se mide aparte
    EVENTOS = {
        "buen_fin": "is_buen_fin",
        "quincena": "is_payday",
        "navidad": "is_navidad_season",
        "semana_santa": "is_semana_santa",
    }
    DIAS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    # Cifras de la exploración previa (CLAUDE.md), solo para comparar
    REFERENCIA = {
        "buen_fin": 2.6,
        "quincena": 1.4,
        "fin_de_semana": 1.5,
        "navidad": 1.16,
        "semana_santa": 0.64,
        "wape_sistema_pct": 18.6,
    }
    return (
        DIAS_ES,
        EVENTOS,
        FACTOR_EXPLORACION,
        HORIZONTE,
        INICIO_EVALUACION,
        REFERENCIA,
        UMBRAL_ATIPICO,
    )


@app.cell
def _(Path, get_config, mo):
    _args = mo.cli_args()
    _cfg = get_config()
    ruta_panel = Path(str(_args.get("panel") or _cfg.interim_dir / "panel.parquet"))
    ruta_reporte = Path(str(_args.get("reporte") or _cfg.artifacts_dir / "calidad" / "reporte.json"))
    ruta_salida = Path(str(_args.get("salida") or _cfg.artifacts_dir / "eda" / "resumen.json"))
    return ruta_panel, ruta_reporte, ruta_salida


@app.cell
def _(LLAVE, SERIE, json, pl, ruta_panel, ruta_reporte, sha256):
    fuente_eda = {
        "panel_sha256": sha256(ruta_panel),
        "reporte_calidad_sha256": sha256(ruta_reporte),
    }
    reporte = json.loads(ruta_reporte.read_text())
    panel = (
        pl.read_parquet(ruta_panel)
        .sort([*SERIE, "date"])
        .with_columns(
            pl.col("units_sold").cast(pl.Float64).alias("u"),
            pl.col("date").dt.truncate("1mo").alias("mes"),
        )
    )
    # Los desplazamientos por serie de las secciones 3 y 5 suponen la malla completa
    _esperadas = panel["date"].n_unique() * panel.select(SERIE).n_unique()
    assert panel.height == _esperadas, "el panel no es la malla completa"
    assert not panel.select(LLAVE).is_duplicated().any(), "el panel repite la llave"
    fuente_eda["datos_crudos_sha256"] = reporte.get("fuente", {})
    return fuente_eda, panel, reporte


@app.cell
def _(mo):
    mo.md(r"""
    # EDA · demanda por tienda × categoría

    Evidencia de los patrones que guían el modelado. Cada afirmación cita una cifra calculada aquí; las cifras
    clave quedan en `artifacts/eda/resumen.json` para los agentes de E7.

    1. Estructura y calidad · 2. Demanda y calendario · 3. Promociones · 4. Tiendas ·
    5. Sistema actual contra baselines · 6. Implicaciones para el modelado
    """)
    return


@app.cell
def _(mo, reporte, tabla_md):
    _d = reporte["dimensiones"]
    _m = reporte["malla"]
    _c = reporte["conteos"]
    _n = reporte["nulos"]
    _f = reporte["fuga_replenishment_signal"]
    _v = _f["mejor_ventana"]
    resumen_calidad = {
        "dimensiones": {
            k: _d[k] for k in ("filas", "tiendas", "categorias", "dias", "series", "desde", "hasta")
        },
        "malla": {"esperadas": _m["esperadas"], "faltantes": _m["faltantes"], "bloques": len(_m["bloques"])},
        "conteos_inconsistentes_pct": _c["pct_inconsistentes"],
        "avg_ticket_sobreestimacion_pct": _c["avg_ticket"]["sobreestimacion_media_pct_en_inconsistentes"],
        "nulos_pct": {col: v["pct"] for col, v in _n.items()},
        "fuga_replenishment_signal": {
            "ventana_desde": _v["desde"],
            "ventana_hasta": _v["hasta"],
            "corr_intraserie": _v["corr"],
            "factor": _f["factor"],
            "rezago_minimo_desde_el_origen": _f["rezago_minimo_desde_el_origen"],
        },
    }
    _bloques = ", ".join(f"{b['store_id']} ({b['desde']} a {b['hasta']})" for b in _m["bloques"]) or "ninguno"
    _tabla_nulos = tabla_md(
        ["columna", "% nulos", "patrón"], [(col, f"{v['pct']:.2f}", v["patron"]) for col, v in _n.items()]
    )
    mo.md(f"""
    ## 1 · Estructura y calidad

    Resumen del reporte de E1 (`artifacts/calidad/reporte.json`); aquí no se recalcula.

    - **Dimensiones:** {_d["filas"]:,} filas, {_d["tiendas"]} tiendas × {_d["categorias"]} categorías =
      {_d["series"]} series, {_d["dias"]} días ({_d["desde"]} a {_d["hasta"]}).
    - **Malla:** {_m["esperadas"]:,} filas esperadas; faltan {_m["faltantes"]} en
      {len(_m["bloques"])} bloques: {_bloques}. El panel las conserva con `flag_fila_faltante`.
    - **Conteos:** {_c["pct_inconsistentes"]:.2f} % de las filas tiene `cash + card < total`; ahí `avg_ticket`
      sobreestima el ticket en {_c["avg_ticket"]["sobreestimacion_media_pct_en_inconsistentes"]:.1f} %.
    - **Fuga:** `replenishment_signal` ≈ {_f["factor"]} × la media de `units_sold` de t{_v["desde"]:+d} a
      t{_v["hasta"]:+d} (correlación intraserie {_v["corr"]}). Con origen T solo se usa la señal de
      T−{_f["rezago_minimo_desde_el_origen"]} o antes.

    {_tabla_nulos}
    """)
    return (resumen_calidad,)


@app.cell
def _(EVENTOS, SERIE, UMBRAL_ATIPICO, panel, pl):
    # Días atípicos: índice = suma de unidades observadas / suma de las medianas de esas mismas series
    _obs = panel.filter(pl.col("u").is_not_null()).with_columns(pl.col("u").median().over(SERIE).alias("med"))
    _indice = (pl.col("u").sum() / pl.col("med").sum()).alias("indice")
    _por_tienda = _obs.group_by("store_id", "date").agg(_indice)
    indice_diario = (
        _obs.group_by("date")
        .agg(_indice)
        .join(
            _por_tienda.group_by("date").agg(
                ((pl.col("indice") < UMBRAL_ATIPICO).mean() * 100).alias("pct_tiendas_bajo_umbral")
            ),
            on="date",
        )
        .sort("date")
    )
    fechas_atipicas = indice_diario.filter(pl.col("indice") < UMBRAL_ATIPICO)["date"]

    # Días limpios: sin eventos, festivos ni atípicos. Son la base contra la que se mide cada efecto
    _algun_evento = pl.any_horizontal(*(pl.col(c) for c in EVENTOS.values()))
    _limpio = (~_algun_evento & ~pl.col("is_holiday") & ~pl.col("atipico") & pl.col("u").is_not_null()).alias(
        "limpio"
    )
    _media_limpia = pl.col("u").filter("limpio").mean()

    cal = (
        panel.with_columns(
            pl.col("date").is_in(fechas_atipicas.implode()).alias("atipico"),
            # Festivo que no forma parte del evento: Nochebuena dentro de Navidad, por ejemplo
            (pl.col("is_holiday") & ~pl.col("is_buen_fin") & ~pl.col("is_semana_santa")).alias(
                "festivo_ajeno"
            ),
        )
        .with_columns(_limpio)
        .with_columns(
            _media_limpia.over([*SERIE, "mes", "day_of_week"]).alias("esperado"),
            pl.col("u")
            .filter(pl.col("limpio") & ~pl.col("is_weekend"))
            .mean()
            .over([*SERIE, "mes"])
            .alias("esperado_habil"),
            _media_limpia.over([*SERIE, "mes"]).alias("esperado_mes"),
            _media_limpia.over([*SERIE, "day_of_week"]).alias("esperado_dia"),
        )
        .with_columns(
            (pl.col("u") / pl.col("esperado")).alias("ratio"),
            (pl.col("u") / pl.col("esperado_habil")).alias("ratio_habil"),
            (pl.col("u") / pl.col("esperado_mes")).alias("ratio_mes"),
            (pl.col("u") / pl.col("esperado_dia")).alias("ratio_dia"),
        )
    )
    return cal, fechas_atipicas, indice_diario


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Demanda y calendario

    Todos los efectos se miden como **cocientes** contra la demanda esperada de la misma serie, porque los
    niveles de las series difieren mucho (ver dispersión) y el calendario actúa en proporción al nivel. Un
    **día limpio** no tiene Buen Fin, quincena, Navidad, Semana Santa, festivo ni es atípico.
    """)
    return


@app.cell
def _(alt, mo, panel, pl, redondear):
    _obs = panel.filter(pl.col("u").is_not_null())
    _cat = _obs.group_by("category").agg(pl.col("u").mean().alias("nivel")).sort("nivel", descending=True)
    _fmt = _obs.group_by("store_format").agg(pl.col("u").mean().alias("nivel")).sort("nivel", descending=True)
    _cruce = _obs.group_by("category", "store_format").agg(pl.col("u").mean().round(0).alias("nivel"))
    _part = (
        _obs.group_by("category").agg(pl.col("u").sum()).with_columns(pl.col("u") / pl.col("u").sum() * 100)
    )
    resumen_nivel = {
        "nivel_medio_por_categoria": {c: redondear(v, 1) for c, v in _cat.iter_rows()},
        "nivel_medio_por_formato": {f: redondear(v, 1) for f, v in _fmt.iter_rows()},
        "participacion_pct_por_categoria": {
            c: redondear(v, 2) for c, v in _part.sort("category").iter_rows()
        },
        "razon_categoria_max_min": redondear(_cat["nivel"].max() / _cat["nivel"].min(), 2),
        "razon_formato_max_min": redondear(_fmt["nivel"].max() / _fmt["nivel"].min(), 2),
    }
    _titulo = (
        f"{_cat['category'][0]} vende {resumen_nivel['razon_categoria_max_min']:.1f}× lo que "
        f"{_cat['category'][-1]}; un {_fmt['store_format'][0]} vende "
        f"{resumen_nivel['razon_formato_max_min']:.1f}× un {_fmt['store_format'][-1]}"
    )
    _base = alt.Chart(_cruce).encode(
        x=alt.X("store_format:N", title="formato", sort=_fmt["store_format"].to_list()),
        y=alt.Y("category:N", title="categoría", sort=_cat["category"].to_list()),
    )
    _grafica = (
        _base.mark_rect().encode(
            color=alt.Color("nivel:Q", title="unidades/día", scale=alt.Scale(type="log"))
        )
        + _base.mark_text(fontSize=11).encode(text=alt.Text("nivel:Q", format=",.0f"))
    ).properties(title=_titulo, width=360, height=220)
    mo.vstack(
        [
            mo.md("### Nivel por categoría y formato\nUnidades medias por serie y día (filas observadas)."),
            _grafica,
        ]
    )
    return (resumen_nivel,)


@app.cell
def _(alt, cal, cifra, mo, pl, redondear):
    # Semanas completas (7 fechas), para que la mezcla de días de la semana no mueva la media
    _semanal = (
        cal.filter(pl.col("u").is_not_null())
        .group_by(pl.col("date").dt.truncate("1w").alias("semana"))
        .agg(pl.col("u").mean().alias("unidades"), pl.col("date").n_unique().alias("dias"))
        .filter(pl.col("dias") == 7)
        .sort("semana")
    )
    # Índice mensual en días limpios, ajustado por día de la semana y nivel de cada serie
    _mensual = (
        cal.filter(pl.col("limpio") & pl.col("ratio_dia").is_not_null())
        .group_by("mes")
        .agg(pl.col("ratio_dia").mean().alias("indice"))
        .sort("mes")
    )
    _pares = _mensual.with_columns(pl.col("mes").dt.offset_by("1y").alias("mes_siguiente")).join(
        _mensual, left_on="mes_siguiente", right_on="mes", suffix="_siguiente"
    )
    _yoy = (
        redondear(((_pares["indice_siguiente"] / _pares["indice"]).mean() - 1) * 100, 2)
        if _pares.height
        else None
    )
    _alto = _mensual.sort("indice", descending=True).row(0, named=True) if _mensual.height else None
    _bajo = _mensual.sort("indice").row(0, named=True) if _mensual.height else None

    def _mes(fila):
        return (
            {"mes": fila["mes"].strftime("%Y-%m"), "indice": redondear(fila["indice"], 3)} if fila else None
        )

    resumen_tendencia = {
        "meses_en_periodo": _mensual.height,
        "crecimiento_interanual_pct": _yoy,
        "meses_comparados": [m.strftime("%Y-%m") for m in _pares["mes_siguiente"]],
        "mes_mas_alto": _mes(_alto),
        "mes_mas_bajo": _mes(_bajo),
        "razon_mes_max_min": redondear(_alto["indice"] / _bajo["indice"], 3) if _alto and _bajo else None,
        "razon_semana_max_mediana": redondear(_semanal["unidades"].max() / _semanal["unidades"].median(), 2)
        if _semanal.height
        else None,
    }
    _alto_txt = resumen_tendencia["mes_mas_alto"]["mes"] if _alto else "s/d"
    _bajo_txt = resumen_tendencia["mes_mas_bajo"]["mes"] if _bajo else "s/d"
    _razon_txt = cifra(resumen_tendencia["razon_mes_max_min"], ".2f")
    _titulo = (
        f"La semana más alta vende {cifra(resumen_tendencia['razon_semana_max_mediana'], '.1f')}× la semana "
        f"mediana; el crecimiento interanual es {cifra(_yoy, '+.1f', ' %')}"
    )
    _g_semanal = (
        alt.Chart(_semanal)
        .mark_line(point=True)
        .encode(
            x=alt.X("semana:T", title="semana"),
            y=alt.Y("unidades:Q", title="unidades medias por serie y día"),
            tooltip=["semana:T", alt.Tooltip("unidades:Q", format=",.0f")],
        )
        .properties(title=_titulo, width=640, height=220)
    )
    _g_mensual = (
        alt.Chart(_mensual)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(mes):O", title="mes"),
            y=alt.Y("indice:Q", title="índice (días limpios)"),
            tooltip=[alt.Tooltip("yearmonth(mes):O", title="mes"), alt.Tooltip("indice:Q", format=".3f")],
        )
        .properties(
            title=(
                f"Aun sin eventos, {_alto_txt} vende {_razon_txt}× lo que {_bajo_txt}: el nivel cambia de un "
                "mes a otro"
            ),
            width=640,
        )
    )
    mo.vstack(
        [
            mo.md(
                "### Tendencia en el periodo\nArriba, unidades medias por serie y día en semanas completas "
                "(incluye eventos). Abajo, el índice mensual en días limpios: `units / media de la serie en "
                "ese día de la semana`. Crecimiento interanual en los meses que se repiten "
                f"({', '.join(resumen_tendencia['meses_comparados']) or 'ninguno'}): "
                f"**{cifra(_yoy, '+.2f', ' %')}**."
            ),
            _g_semanal,
            _g_mensual,
        ]
    )
    return (resumen_tendencia,)


@app.cell
def _(DIAS_ES, alt, cal, cifra, mo, pl, redondear):
    _perfil = (
        cal.filter(pl.col("limpio") & pl.col("ratio_mes").is_not_null())
        .group_by("day_of_week")
        .agg(pl.col("ratio_mes").mean().alias("factor"))
        .sort("day_of_week")
        .with_columns(
            pl.col("day_of_week").map_elements(lambda d: DIAS_ES[d], return_dtype=pl.String).alias("dia")
        )
    )
    _alto = _perfil.sort("factor", descending=True).row(0, named=True)
    _bajo = _perfil.sort("factor").row(0, named=True)
    resumen_semana = {
        "perfil_dia_semana": {r["dia"]: redondear(r["factor"], 3) for r in _perfil.iter_rows(named=True)},
        "razon_dia_max_min": redondear(_alto["factor"] / _bajo["factor"], 3),
        "dia_max": _alto["dia"],
        "dia_min": _bajo["dia"],
    }
    _vie = resumen_semana["perfil_dia_semana"].get("viernes")
    _grafica = (
        alt.Chart(_perfil)
        .mark_bar()
        .encode(
            x=alt.X("dia:N", sort=list(DIAS_ES), title=None),
            y=alt.Y("factor:Q", title="unidades / media del mes"),
            tooltip=["dia:N", alt.Tooltip("factor:Q", format=".3f")],
        )
        .properties(
            title=(
                f"El {_alto['dia']} vende {resumen_semana['razon_dia_max_min']:.2f}× el {_bajo['dia']}; el "
                f"viernes ({cifra(_vie, '.2f')}) ya sube: no es solo fin de semana"
            ),
            width=420,
            height=220,
        )
    )
    mo.vstack(
        [
            mo.md(
                "### Estacionalidad semanal\nMedia, en días limpios, de `units / media de la serie en los "
                "días limpios del mismo mes`."
            ),
            _grafica,
        ]
    )
    return (resumen_semana,)


@app.cell
def _(SERIE, alt, cal, mo, pl, redondear):
    _series = cal.group_by(*SERIE).agg(
        pl.col("u").mean().alias("nivel"),
        pl.col("ratio").filter("limpio").std().alias("cv_residual"),
    )
    _q = {
        f"p{int(q * 100):02d}": redondear(_series["nivel"].quantile(q), 1)
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    resumen_dispersion = {
        "nivel_serie_cuantiles": _q,
        "razon_serie_max_min": redondear(_series["nivel"].max() / _series["nivel"].min(), 1),
        "cv_entre_series": redondear(_series["nivel"].std() / _series["nivel"].mean(), 3),
        "cv_residual_mediano": redondear(_series["cv_residual"].median(), 3),
    }
    _grafica = (
        alt.Chart(_series)
        .mark_boxplot()
        .encode(
            x=alt.X("category:N", title="categoría"),
            y=alt.Y("nivel:Q", title="unidades medias por día (escala log)", scale=alt.Scale(type="log")),
        )
        .properties(
            title=(
                f"La serie más grande vende {resumen_dispersion['razon_serie_max_min']:.0f}× la más chica; "
                "el ruido diario sin calendario es "
                f"~{100 * (resumen_dispersion['cv_residual_mediano'] or 0):.0f} %"
            ),
            width=480,
            height=240,
        )
    )
    mo.vstack(
        [
            mo.md(
                "### Dispersión entre series\nCada caja resume las 80 series de la categoría (una por "
                "tienda). El ruido residual es la desviación estándar del cociente contra lo esperado en "
                "días limpios, mediana entre series."
            ),
            _grafica,
        ]
    )
    return (resumen_dispersion,)


@app.cell
def _(EVENTOS, REFERENCIA, alt, cal, cifra, mo, pl, redondear, tabla_md):
    def _media(df, col):
        return df[col].mean() if df.height else None

    def _fin_de_semana(df):
        num, den = (
            _media(df.filter("is_weekend"), "ratio_habil"),
            _media(df.filter(~pl.col("is_weekend")), "ratio_habil"),
        )
        return num / den if num is not None and den else None

    def _dias_del_evento(nombre):
        otros = [pl.col(c) for k, c in EVENTOS.items() if k != nombre]
        return cal.filter(
            pl.col(EVENTOS[nombre])
            & ~pl.any_horizontal(*otros)
            & ~pl.col("festivo_ajeno")
            & ~pl.col("atipico")
            & pl.col("ratio").is_not_null()
        )

    _categorias = sorted(cal["category"].unique().to_list())
    _habiles = cal.filter(pl.col("limpio") & pl.col("ratio_habil").is_not_null())
    _filas = {k: _dias_del_evento(k) for k in EVENTOS}

    _global = {k: _media(df, "ratio") for k, df in _filas.items()}
    _global["fin_de_semana"] = _fin_de_semana(_habiles)
    _por_cat = {
        k: {c: _media(df.filter(pl.col("category") == c), "ratio") for c in _categorias}
        for k, df in _filas.items()
    }
    _por_cat["fin_de_semana"] = {
        c: _fin_de_semana(_habiles.filter(pl.col("category") == c)) for c in _categorias
    }
    _dias = {k: df["date"].n_unique() for k, df in _filas.items()}
    _dias["fin_de_semana"] = _habiles.filter("is_weekend")["date"].n_unique()

    _orden = ["buen_fin", "quincena", "fin_de_semana", "navidad", "semana_santa"]
    efectos_calendario = {k: redondear(_global[k]) for k in _orden}
    efectos_por_categoria = {k: {c: redondear(v) for c, v in _por_cat[k].items()} for k in _orden}

    def _dispersion(valores):
        v = [x for x in valores.values() if x is not None]
        if not v:
            return None
        return {"min": min(v), "max": max(v), "rango_pct": redondear((max(v) / min(v) - 1) * 100, 2)}

    resumen_efectos = {
        "efectos_calendario": efectos_calendario,
        "efectos_calendario_por_categoria": efectos_por_categoria,
        "dispersion_efectos_por_categoria": {k: _dispersion(efectos_por_categoria[k]) for k in _orden},
        "dias_por_efecto": {k: _dias[k] for k in _orden},
    }
    _disp = {k: d for k, d in resumen_efectos["dispersion_efectos_por_categoria"].items() if d}
    _peor = max(_disp, key=lambda k: _disp[k]["rango_pct"]) if _disp else None
    _rango_max = f"{_disp[_peor]['rango_pct']:.1f} % en {_peor}, con {_dias[_peor]} días" if _peor else "s/d"

    _largo = pl.DataFrame(
        [
            {"efecto": k, "categoria": c, "multiplicador": v}
            for k in _orden
            for c, v in efectos_por_categoria[k].items()
            if v is not None
        ],
        schema={"efecto": pl.String, "categoria": pl.String, "multiplicador": pl.Float64},
    )
    _grafica = (
        alt.Chart(_largo)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("multiplicador:Q", scale=alt.Scale(type="log"), title="multiplicador (escala log)"),
            y=alt.Y("efecto:N", sort=_orden, title=None),
            color=alt.Color("categoria:N", title="categoría"),
            tooltip=["efecto:N", "categoria:N", alt.Tooltip("multiplicador:Q", format=".3f")],
        )
        .properties(
            title=(
                "Los efectos de calendario son multiplicativos y casi iguales entre categorías (rango "
                f"máximo: {_rango_max})"
            ),
            width=520,
            height=200,
        )
    )
    _tabla = tabla_md(
        ["efecto", "multiplicador", "exploración", "días", "rango entre categorías"],
        [
            (
                k,
                cifra(efectos_calendario[k], ".3f"),
                REFERENCIA[k],
                _dias[k],
                cifra(
                    (resumen_efectos["dispersion_efectos_por_categoria"][k] or {}).get("rango_pct"),
                    ".1f",
                    " %",
                ),
            )
            for k in _orden
        ],
    )
    mo.vstack(
        [
            mo.md(
                "### Efectos de calendario\nMultiplicador = media de `units / esperado` en los días del "
                "evento, donde *esperado* es la media de la misma serie, el mismo mes y el mismo día de la "
                "semana en días limpios. Así la quincena no se confunde con el fin de semana ni con el nivel "
                "del mes. Se excluyen los días con otro evento, con un festivo ajeno o atípicos. El fin de "
                "semana se mide contra los días hábiles limpios de la serie en el mes. `s/d`: el evento no "
                "tiene días en el periodo.\n\n" + _tabla
            ),
            _grafica,
        ]
    )
    return efectos_calendario, resumen_efectos


@app.cell
def _(alt, cal, mo, pl, redondear):
    _festivos = (
        cal.filter(pl.col("festivo_ajeno") & pl.col("ratio").is_not_null())
        .group_by("date", "holiday_name")
        .agg(
            pl.col("ratio").mean().alias("multiplicador"),
            pl.col("is_navidad_season").first().alias("en_navidad"),
            pl.col("is_payday").first().alias("en_quincena"),
        )
        .sort("date")
    )
    resumen_festivos = [
        {
            "fecha": str(r["date"]),
            "nombre": r["holiday_name"],
            "multiplicador": redondear(r["multiplicador"], 3),
            "en_navidad": r["en_navidad"],
            "en_quincena": r["en_quincena"],
        }
        for r in _festivos.iter_rows(named=True)
    ]
    _grafica = (
        alt.Chart(_festivos.with_columns(pl.format("{} · {}", "date", "holiday_name").alias("festivo")))
        .mark_bar()
        .encode(
            x=alt.X("multiplicador:Q", scale=alt.Scale(type="log"), title="multiplicador (escala log)"),
            y=alt.Y("festivo:N", sort=None, title=None),
            color=alt.condition(alt.datum.multiplicador > 1, alt.value("#4c78a8"), alt.value("#e45756")),
            tooltip=["festivo:N", alt.Tooltip("multiplicador:Q", format=".3f")],
        )
        .properties(
            title="Cada festivo tiene su propio efecto: unos suben la demanda y otros la hunden", width=420
        )
    )
    mo.vstack(
        [
            mo.md(
                "### Festivos fuera de los eventos\nMismo cociente que los efectos de calendario, por fecha. "
                "Los que caen en Navidad o en quincena incluyen ese efecto también."
            ),
            _grafica if _festivos.height else mo.md("_No hay festivos con datos en este periodo._"),
        ]
    )
    return (resumen_festivos,)


@app.cell
def _(UMBRAL_ATIPICO, alt, fechas_atipicas, indice_diario, mo, pl, redondear):
    _atip = indice_diario.filter(pl.col("date").is_in(fechas_atipicas.implode()))
    _resto = indice_diario.filter(~pl.col("date").is_in(fechas_atipicas.implode())).sort("indice")
    resumen_atipicos = {
        "umbral": UMBRAL_ATIPICO,
        "fechas": [str(d) for d in fechas_atipicas],
        "detalle": [
            {
                "fecha": str(r["date"]),
                "indice": redondear(r["indice"], 3),
                "pct_tiendas_bajo_umbral": redondear(r["pct_tiendas_bajo_umbral"], 1),
            }
            for r in _atip.iter_rows(named=True)
        ],
        "siguiente_mas_bajo": {"fecha": str(_resto["date"][0]), "indice": redondear(_resto["indice"][0], 3)}
        if _resto.height
        else None,
    }
    _lista = ", ".join(resumen_atipicos["fechas"]) or "ninguna fecha"
    _grafica = (
        alt.Chart(indice_diario)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("indice:Q", title="unidades / mediana de cada serie"),
            tooltip=["date:T", alt.Tooltip("indice:Q", format=".3f")],
        )
        + alt.Chart(pl.DataFrame({"umbral": [UMBRAL_ATIPICO]}))
        .mark_rule(color="#e45756")
        .encode(y="umbral:Q")
        + alt.Chart(_atip).mark_point(color="#e45756", size=80, filled=True).encode(x="date:T", y="indice:Q")
    ).properties(
        title=f"Solo {_lista} cae debajo del {UMBRAL_ATIPICO:.0%} de la mediana", width=640, height=200
    )
    _detalle = "; ".join(
        f"{d['fecha']}: índice {d['indice']}, {d['pct_tiendas_bajo_umbral']} % de las tiendas bajo el umbral"
        for d in resumen_atipicos["detalle"]
    )
    mo.vstack(
        [
            mo.md(
                "### Días atípicos\nÍndice diario = suma de las unidades observadas / suma de las medianas "
                f"de esas series. Un día es atípico si el índice queda debajo de {UMBRAL_ATIPICO}. "
                f"{_detalle or 'No hay días atípicos.'}"
            ),
            _grafica,
        ]
    )
    return (resumen_atipicos,)


@app.cell
def _(DIAS_ES, SERIE, alt, mo, panel, pl, redondear):
    _base = panel.filter(~pl.col("flag_fila_faltante")).with_columns(
        # Nivel previo: media de t−7 a t−1 entre la media de la serie (¿se promueve cuando la demanda cae?)
        (
            pl.col("u").shift(1).rolling_mean(7, min_samples=1).over(SERIE) / pl.col("u").mean().over(SERIE)
        ).alias("nivel_previo")
    )
    _base = _base.with_columns(
        # Quintil por rango (1 a 5); los nulos del inicio de cada serie quedan fuera
        pl.concat_str(
            pl.lit("Q"),
            ((pl.col("nivel_previo").rank("ordinal") - 1) * 5 // pl.col("nivel_previo").count() + 1).cast(
                pl.String
            ),
        ).alias("quintil"),
        pl.col("day_of_week").map_elements(lambda d: DIAS_ES[d], return_dtype=pl.String).alias("dia"),
        pl.col("month").cast(pl.String).str.zfill(2).alias("mes_num"),
    )
    tasa_global = _base["has_promotion"].mean()
    _cortes = {
        "formato": "store_format",
        "dia_semana": "dia",
        "mes": "mes_num",
        "quintil_nivel_previo": "quintil",
    }
    _orden_dia = {d: i for i, d in enumerate(DIAS_ES)}
    _tasas = {}
    _largo = []
    for _nombre, _col in _cortes.items():
        _t = (
            _base.filter(pl.col(_col).is_not_null())
            .group_by(_col)
            .agg((pl.col("has_promotion").mean() * 100).alias("tasa"))
        )
        _t = _t.sort(pl.col(_col).replace_strict(_orden_dia, default=None) if _col == "dia" else pl.col(_col))
        _tasas[_nombre] = {v: redondear(t, 2) for v, t in _t.iter_rows()}
        _largo += [{"corte": _nombre, "valor": v, "tasa": t} for v, t in _t.iter_rows()]
    _rangos = {k: redondear(max(v.values()) - min(v.values()), 2) for k, v in _tasas.items() if v}

    # Autocorrelación de la asignación dentro de cada serie (centrada por serie)
    _ac = (
        panel.select(
            *SERIE,
            pl.col("has_promotion").cast(pl.Float64).alias("p"),
            pl.col("has_promotion").shift(1).over(SERIE).cast(pl.Float64).alias("l1"),
            pl.col("has_promotion").shift(7).over(SERIE).cast(pl.Float64).alias("l7"),
        )
        .drop_nulls()
        .with_columns(pl.col(c) - pl.col(c).mean().over(SERIE) for c in ("p", "l1", "l7"))
    )
    _cond = (
        panel.select(
            pl.col("has_promotion").alias("hoy"), pl.col("has_promotion").shift(1).over(SERIE).alias("ayer")
        )
        .drop_nulls()
        .group_by("ayer")
        .agg((pl.col("hoy").mean() * 100).alias("tasa"))
    )
    _cond_d = dict(_cond.iter_rows())

    # Lift ingenuo: por serie, media con promoción / sin promoción − 1; sin controlar calendario
    _lift_serie = (
        panel.filter(pl.col("u").is_not_null())
        .group_by(*SERIE)
        .agg(
            (
                pl.col("u").filter(pl.col("has_promotion") == 1).mean()
                / pl.col("u").filter(pl.col("has_promotion") == 0).mean()
                - 1
            ).alias("lift")
        )
        .drop_nulls()
    )
    _lift = (
        _lift_serie.group_by("category")
        .agg(
            (pl.col("lift").mean() * 100).alias("lift"),
            (1.96 * pl.col("lift").std() / pl.col("lift").count().sqrt() * 100).alias("me"),
            pl.len().alias("series"),
        )
        .with_columns(
            (pl.col("lift") - pl.col("me")).alias("lo"), (pl.col("lift") + pl.col("me")).alias("hi")
        )
        .sort("category")
    )
    _excluyen_cero = _lift.filter((pl.col("lo") > 0) | (pl.col("hi") < 0))["category"].to_list()
    resumen_promociones = {
        "tasa_global_pct": redondear(tasa_global * 100, 2),
        "tasa_pct": _tasas,
        "rango_tasa_pp": _rangos,
        "autocorrelacion": {
            "lag_1": redondear(_ac.select(pl.corr("p", "l1")).item(), 4),
            "lag_7": redondear(_ac.select(pl.corr("p", "l7")).item(), 4),
            "tasa_pct_si_ayer_hubo": redondear(_cond_d.get(1), 2),
            "tasa_pct_si_ayer_no_hubo": redondear(_cond_d.get(0), 2),
        },
        "lift_ingenuo_pct": {r["category"]: redondear(r["lift"], 2) for r in _lift.iter_rows(named=True)},
        "lift_ingenuo_ic95_pct": {
            r["category"]: [redondear(r["lo"], 2), redondear(r["hi"], 2)] for r in _lift.iter_rows(named=True)
        },
        "categorias_con_ic95_sin_cero": _excluyen_cero,
        "nota": "Revisión descriptiva; la estimación formal del efecto es de E6.",
    }

    _df_tasas = pl.DataFrame(_largo, schema={"corte": pl.String, "valor": pl.String, "tasa": pl.Float64})
    _g_tasas = alt.hconcat(
        *(
            alt.Chart(_df_tasas.filter(pl.col("corte") == _c))
            .mark_bar()
            .encode(
                x=alt.X("valor:N", sort=None, title=_c),
                y=alt.Y("tasa:Q", title="% de días con promoción", scale=alt.Scale(domain=[0, 30])),
                tooltip=["valor:N", alt.Tooltip("tasa:Q", format=".2f")],
            )
            .properties(width=150, height=180)
            + alt.Chart(pl.DataFrame({"t": [tasa_global * 100]})).mark_rule(strokeDash=[4, 4]).encode(y="t:Q")
            for _c in _cortes
        )
    ).properties(
        title=(
            f"La asignación es plana: {tasa_global * 100:.1f} % en todos los cortes (rango máximo "
            f"{max(_rangos.values()):.1f} pp)"
        )
    )
    _g_lift = (
        alt.Chart(_lift)
        .mark_errorbar()
        .encode(
            x=alt.X("lo:Q", title="lift ingenuo (%) e IC 95 %"), x2="hi:Q", y=alt.Y("category:N", title=None)
        )
        + alt.Chart(_lift).mark_point(filled=True, size=60).encode(x="lift:Q", y="category:N")
        + alt.Chart(pl.DataFrame({"cero": [0.0]})).mark_rule(color="gray").encode(x="cero:Q")
    ).properties(
        title=(
            f"El lift ingenuo es pequeño (de {_lift['lift'].min():+.1f} % a {_lift['lift'].max():+.1f} %); "
            f"IC sin cero: {', '.join(_excluyen_cero) or 'ninguna categoría'}"
        ),
        width=420,
        height=180,
    )
    mo.vstack(
        [
            mo.md(
                "## 3 · Promociones\nRevisión descriptiva: ¿la asignación parece aleatoria y qué tan grande "
                "es el efecto bruto? La estimación formal es de E6.\n\n- Autocorrelación de la asignación "
                f"dentro de la serie: lag 1 = {resumen_promociones['autocorrelacion']['lag_1']}, lag 7 = "
                f"{resumen_promociones['autocorrelacion']['lag_7']}. Tasa si ayer hubo promoción: "
                f"{resumen_promociones['autocorrelacion']['tasa_pct_si_ayer_hubo']} %; si no hubo: "
                f"{resumen_promociones['autocorrelacion']['tasa_pct_si_ayer_no_hubo']} %.\n- *Nivel previo*: "
                "media de `units_sold` de t−7 a t−1 relativa a la media de la serie, en quintiles. Si la "
                "promoción respondiera a caídas de demanda, Q1 tendría más promociones."
            ),
            _g_tasas,
            _g_lift,
        ]
    )
    return (resumen_promociones,)


@app.cell
def _(SERIE, alt, cifra, mo, panel, pl, redondear):
    _attrs = ["store_format", "region", "socioeconomic_level", "size_sqm", "num_checkouts"]
    _tiendas = (
        panel.filter(pl.col("u").is_not_null())
        .group_by(*SERIE)
        .agg(pl.col("u").mean().alias("nivel"), *(pl.col(a).first() for a in _attrs))
        .group_by("store_id")
        .agg(pl.col("nivel").sum(), *(pl.col(a).first() for a in _attrs))
    )

    def _conteo(col):
        return {k: n for k, n in _tiendas.group_by(col).len().sort(col).iter_rows()}

    def _corr(df, a, b, metodo="pearson"):
        return redondear(df.select(pl.corr(a, b, method=metodo)).item(), 3) if df.height >= 3 else None

    _por_formato = {
        f: {
            "tiendas": d.height,
            "corr_nivel_m2": _corr(d, "nivel", "size_sqm"),
            "unidades_por_1000_m2": redondear((d["nivel"] / d["size_sqm"] * 1000).mean(), 1),
        }
        for (f,), d in sorted(_tiendas.partition_by("store_format", as_dict=True).items())
    }
    _cruce = _tiendas.group_by("store_format", "region").len().sort("store_format", "region")
    _cruce_d = {}
    for _f, _r, _n in _cruce.iter_rows():
        _cruce_d.setdefault(_f, {})[_r] = _n
    resumen_tiendas = {
        "tiendas": _tiendas.height,
        "por_formato": _conteo("store_format"),
        "por_region": _conteo("region"),
        "por_nse": _conteo("socioeconomic_level"),
        "formato_por_region": _cruce_d,
        "corr_nivel_m2": _corr(_tiendas, "nivel", "size_sqm"),
        "corr_nivel_cajas": _corr(_tiendas, "nivel", "num_checkouts"),
        "corr_nivel_m2_spearman": _corr(_tiendas, "nivel", "size_sqm", "spearman"),
        "corr_m2_cajas": _corr(_tiendas, "size_sqm", "num_checkouts"),
        "por_formato_detalle": _por_formato,
        "nivel_medio_tienda_por_nse": {
            k: redondear(v, 1)
            for k, v in _tiendas.group_by("socioeconomic_level")
            .agg(pl.col("nivel").mean())
            .sort("socioeconomic_level")
            .iter_rows()
        },
    }
    _r = cifra(resumen_tiendas["corr_nivel_m2"], ".2f")
    _r_cajas = cifra(resumen_tiendas["corr_nivel_cajas"], ".2f")
    _grafica = (
        alt.Chart(_tiendas)
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x=alt.X("size_sqm:Q", title="tamaño (m²)"),
            y=alt.Y("nivel:Q", title="unidades por día (todas las categorías)"),
            color=alt.Color("store_format:N", title="formato"),
            tooltip=[
                "store_id:N",
                "store_format:N",
                "region:N",
                "size_sqm:Q",
                "num_checkouts:Q",
                alt.Tooltip("nivel:Q", format=",.0f"),
            ],
        )
        .properties(
            title=(f"El tamaño explica buena parte del nivel: r = {_r} con m² y {_r_cajas} con cajas"),
            width=480,
            height=260,
        )
    )
    _tabla = "\n".join(
        f"- **{f}:** {d['tiendas']} tiendas; correlación nivel–m² dentro del formato {d['corr_nivel_m2']}; "
        f"{d['unidades_por_1000_m2']} unidades/día por cada 1,000 m²."
        for f, d in _por_formato.items()
    )
    mo.vstack(
        [
            mo.md(
                f"## 4 · Tiendas\n{resumen_tiendas['tiendas']} tiendas. Por formato: "
                f"{resumen_tiendas['por_formato']}. Por región: {resumen_tiendas['por_region']}. Por NSE: "
                f"{resumen_tiendas['por_nse']}.\n\nCorrelación entre m² y cajas: "
                f"{resumen_tiendas['corr_m2_cajas']}, así que las dos variables cuentan casi la misma "
                "historia.\n\n" + _tabla
            ),
            _grafica,
        ]
    )
    return (resumen_tiendas,)


@app.cell
def _(
    FACTOR_EXPLORACION,
    HORIZONTE,
    INICIO_EVALUACION,
    REFERENCIA,
    SERIE,
    alt,
    cal,
    cifra,
    mo,
    pl,
    redondear,
    reporte,
    tabla_md,
):
    _factor = reporte["fuga_replenishment_signal"]["factor"]
    _u = pl.col("u")
    _futuro = [_u.shift(-k).over(SERIE) for k in range(1, HORIZONTE + 1)]

    def _en_ventana(col):
        return pl.any_horizontal(
            pl.col(col).shift(-k).over(SERIE) for k in range(1, HORIZONTE + 1)
        ).fill_null(False)

    _senal_t = pl.col("replenishment_signal")
    _senal_t1 = pl.col("replenishment_signal").shift(1).over(SERIE)
    _variantes = {
        "senal_T_factor_053": _senal_t / FACTOR_EXPLORACION * HORIZONTE,
        "senal_T_factor_ajustado": _senal_t / _factor * HORIZONTE,
        "senal_T_menos_1_factor_053": _senal_t1 / FACTOR_EXPLORACION * HORIZONTE,
        "senal_T_menos_1_factor_ajustado": _senal_t1 / _factor * HORIZONTE,
        "ma7": _u.rolling_mean(7, min_samples=1).over(SERIE) * HORIZONTE,
    }
    # Todo se desplaza sobre la malla completa, ordenada por serie y fecha; se filtra después
    _todo = cal.sort([*SERIE, "date"]).with_columns(
        pl.sum_horizontal(_futuro).alias("fut7"),
        pl.all_horizontal(x.is_not_null() for x in _futuro).alias("fut7_completo"),
        *(expr.alias(k) for k, expr in _variantes.items()),
        pl.when(_en_ventana("is_buen_fin"))
        .then(pl.lit("buen_fin"))
        .when(_en_ventana("is_navidad_season"))
        .then(pl.lit("navidad"))
        .otherwise(pl.lit("resto"))
        .alias("segmento"),
    )
    _ultimo = _todo["date"].max()
    _posibles = _todo.filter(
        (pl.col("date") >= INICIO_EVALUACION) & (pl.col("date") <= _ultimo - pl.duration(days=HORIZONTE))
    )
    _con_fut = _posibles.filter("fut7_completo")
    _eval = _con_fut.drop_nulls(list(_variantes))

    def _wape(df, col):
        total = df["fut7"].sum() if df.height else 0
        return redondear((df[col] - df["fut7"]).abs().sum() / total * 100, 2) if total else None

    def _sesgo(df, col):
        total = df["fut7"].sum() if df.height else 0
        return redondear((df[col] - df["fut7"]).sum() / total * 100, 2) if total else None

    _segmentos = {
        "buen_fin": pl.col("segmento") == "buen_fin",
        "navidad": pl.col("segmento") == "navidad",
        "buen_fin_o_navidad": pl.col("segmento") != "resto",
        "resto": pl.col("segmento") == "resto",
    }
    _wape_seg = {v: {s: _wape(_eval.filter(cond), v) for s, cond in _segmentos.items()} for v in _variantes}
    _w = {v: _wape(_eval, v) for v in _variantes}
    resumen_sistema = {
        "protocolo": {
            "origen": "T = último día observado; se pronostica la suma de units_sold de T+1 a T+7 (fut7)",
            "desde": str(INICIO_EVALUACION),
            "hasta": str(_eval["date"].max()) if _eval.height else None,
            "factor_exploracion": FACTOR_EXPLORACION,
            "factor_ajustado": _factor,
        },
        "origenes": {
            "posibles": _posibles.height,
            "con_fut7_completo": _con_fut.height,
            "evaluados": _eval.height,
            "por_segmento": {s: _eval.filter(c).height for s, c in _segmentos.items()},
        },
        "wape_pct": _w,
        "sesgo_pct": {v: _sesgo(_eval, v) for v in _variantes},
        "wape_por_evento_pct": _wape_seg,
        "aumento_wape_al_quitar_fuga_pp": redondear(
            _w["senal_T_menos_1_factor_ajustado"] - _w["senal_T_factor_053"], 2
        )
        if _w["senal_T_factor_053"] is not None and _w["senal_T_menos_1_factor_ajustado"] is not None
        else None,
        "ventaja_sobre_ma7_sin_fuga_pp": redondear(_w["ma7"] - _w["senal_T_menos_1_factor_ajustado"], 2)
        if _w["ma7"] is not None and _w["senal_T_menos_1_factor_ajustado"] is not None
        else None,
    }

    _largo = pl.DataFrame(
        [
            {"variante": v, "segmento": s, "wape": w}
            for v in _variantes
            for s, w in [("global", _w[v]), *((s, _wape_seg[v][s]) for s in ("buen_fin", "navidad", "resto"))]
            if w is not None
        ],
        schema={"variante": pl.String, "segmento": pl.String, "wape": pl.Float64},
    )
    _grafica = (
        alt.Chart(_largo)
        .mark_bar()
        .encode(
            x=alt.X("segmento:N", sort=["global", "buen_fin", "navidad", "resto"], title=None),
            xOffset=alt.XOffset("variante:N", sort=list(_variantes)),
            y=alt.Y("wape:Q", title="WAPE (%)"),
            color=alt.Color("variante:N", sort=list(_variantes), title="pronóstico"),
            tooltip=["variante:N", "segmento:N", alt.Tooltip("wape:Q", format=".2f")],
        )
        .properties(
            title=(
                f"Sin la fuga, el WAPE del sistema pasa de {cifra(_w['senal_T_factor_053'], '.1f', ' %')} a "
                f"{cifra(_w['senal_T_menos_1_factor_ajustado'], '.1f', ' %')}; la MA7 queda en "
                f"{cifra(_w['ma7'], '.1f', ' %')}"
            ),
            width=520,
            height=260,
        )
    )
    _tabla = tabla_md(
        ["pronóstico", "WAPE", "sesgo", "Buen Fin", "Navidad", "Buen Fin o Navidad", "resto"],
        [
            (
                v,
                cifra(_w[v], ".2f", " %"),
                cifra(resumen_sistema["sesgo_pct"][v], "+.2f", " %"),
                *(
                    cifra(_wape_seg[v][s], ".2f", " %")
                    for s in ("buen_fin", "navidad", "buen_fin_o_navidad", "resto")
                ),
            )
            for v in _variantes
        ],
    )
    _o = resumen_sistema["origenes"]
    mo.vstack(
        [
            mo.md(
                f"## 5 · Sistema actual contra baselines\nOrigen T desde {INICIO_EVALUACION}: se pronostica "
                f"`fut7`, la suma de `units_sold` de T+1 a T+{HORIZONTE}, y solo se evalúan los orígenes con "
                f"los {HORIZONTE} días observados. Sistema actual = `señal / factor × {HORIZONTE}`, con la "
                f"señal fechada en T (como en la exploración) y en T−1, y con el factor {FACTOR_EXPLORACION} "
                f"de la exploración o el ajustado del reporte de E1 ({_factor}). Baseline: MA7 de T−6 a T × "
                f"{HORIZONTE} (para la suma semanal equivale al seasonal naive 7). Los segmentos se asignan "
                "por la ventana objetivo: si contiene un día de Buen Fin es *buen_fin*; si no, y contiene "
                f"uno de Navidad, *navidad*.\n\nOrígenes: {_o['posibles']:,} posibles, "
                f"{_o['con_fut7_completo']:,} con `fut7` completo y {_o['evaluados']:,} con las cinco "
                "predicciones disponibles (la señal es nula en las filas faltantes). Todas las variantes se "
                "miden sobre ese mismo conjunto. Referencia de la exploración: "
                f"{REFERENCIA['wape_sistema_pct']} %. El backtest formal, con horizontes diarios, es de "
                "E3.\n\n" + _tabla
            ),
            _grafica,
        ]
    )
    return (resumen_sistema,)


@app.cell
def _(
    cifra,
    efectos_calendario,
    mo,
    reporte,
    resumen_atipicos,
    resumen_dispersion,
    resumen_efectos,
    resumen_festivos,
    resumen_promociones,
    resumen_semana,
    resumen_sistema,
    resumen_tendencia,
    resumen_tiendas,
):
    _w = resumen_sistema["wape_pct"]
    _sin_fuga = _w["senal_T_menos_1_factor_ajustado"]
    _bf = resumen_sistema["wape_por_evento_pct"]["senal_T_menos_1_factor_ajustado"]["buen_fin"]
    _o = resumen_sistema["origenes"]
    _disp = {k: d for k, d in resumen_efectos["dispersion_efectos_por_categoria"].items() if d}
    _peor = max(_disp, key=lambda k: _disp[k]["rango_pct"]) if _disp else None
    _resto = [d["rango_pct"] for k, d in _disp.items() if k != _peor]
    _perfil = resumen_semana["perfil_dia_semana"]
    _fest = sorted(
        (f for f in resumen_festivos if f["multiplicador"] is not None), key=lambda f: f["multiplicador"]
    )
    _lift = [v for v in resumen_promociones["lift_ingenuo_pct"].values() if v is not None]
    _ac = resumen_promociones["autocorrelacion"]
    _unicos = [k for k in ("buen_fin", "semana_santa") if resumen_efectos["dias_por_efecto"][k]]
    _rezago = reporte["fuga_replenishment_signal"]["rezago_minimo_desde_el_origen"]
    _t = resumen_tendencia
    _atipicos = ", ".join(resumen_atipicos["fechas"]) or "ningún día"

    implicaciones = [
        (
            f"La señal de reposición solo se usa rezagada (T−{_rezago}). Con la señal de T−1 el WAPE del "
            f"sistema sube de {cifra(_w['senal_T_factor_053'], '.2f', ' %')} a "
            f"{cifra(_sin_fuga, '.2f', ' %')}: esa es la cifra honesta contra la que se compara el modelo de "
            "E3–E4."
        ),
        (
            f"La MA7 ({cifra(_w['ma7'], '.2f', ' %')}) queda a "
            f"{cifra(resumen_sistema['ventaja_sobre_ma7_sin_fuga_pp'], '.2f')} pp del sistema sin fuga: el "
            "sistema actual es, en esencia, una media móvil, y el modelo tiene que ganarle a ambos."
        ),
        (
            "El calendario actúa como multiplicador común a todas las categorías: el rango entre categorías "
            f"es de {cifra(max(_resto) if _resto else None, '.1f', ' %')} como máximo, salvo {_peor} "
            f"({cifra(_disp[_peor]['rango_pct'] if _peor else None, '.1f', ' %')}, "
            f"{resumen_efectos['dias_por_efecto'].get(_peor, 0)} días). Conviene un modelo global en escala "
            "log con features de calendario compartidas, no un modelo por categoría."
        ),
        (
            f"El perfil semanal no es binario (lunes {cifra(_perfil.get('lunes'), '.2f')}, viernes "
            f"{cifra(_perfil.get('viernes'), '.2f')}, sábado {cifra(_perfil.get('sábado'), '.2f')}): se usa "
            "el día de la semana completo, y la quincena entra como feature propia "
            f"(×{cifra(efectos_calendario['quincena'], '.2f')})."
        ),
        (
            f"Aun en días limpios, el mes más alto ({(_t['mes_mas_alto'] or {}).get('mes', 's/d')}) vende "
            f"{cifra(_t['razon_mes_max_min'], '.2f')}× el más bajo, y el crecimiento interanual es "
            f"{cifra(_t['crecimiento_interanual_pct'], '+.1f', ' %')}. Con {_t['meses_en_periodo']} meses no "
            "se separa la estacionalidad anual de la tendencia: el modelo necesita features de mes y del "
            "nivel reciente, y la validación debe cubrir noviembre y diciembre."
        ),
        (
            f"Buen Fin (×{cifra(efectos_calendario['buen_fin'], '.2f')}) y Semana Santa "
            f"(×{cifra(efectos_calendario['semana_santa'], '.2f')}) ocurren una sola vez, así que su efecto "
            "no se puede validar fuera de muestra. El WAPE del sistema sin fuga en semanas con Buen Fin es "
            f"de {cifra(_bf, '.1f', ' %')}; estos eventos se reportan aparte."
            if _unicos
            else "Este periodo no tiene Buen Fin ni Semana Santa: sus efectos no se pueden estimar aquí."
        ),
        (
            f"Cada festivo tiene su propio efecto, de ×{_fest[0]['multiplicador']} ({_fest[0]['nombre']}) a "
            f"×{_fest[-1]['multiplicador']} ({_fest[-1]['nombre']}): `is_holiday` sola no basta. Días "
            f"atípicos: {_atipicos}."
            if len(_fest) > 1
            else f"Días atípicos: {_atipicos}."
        ),
        (
            f"Las promociones parecen asignadas al azar (tasa de {resumen_promociones['tasa_global_pct']} %, "
            f"autocorrelación lag 1 de {_ac['lag_1']}) y su lift ingenuo es pequeño (máximo "
            f"{cifra(max(_lift) if _lift else None, '+.1f', ' %')}). E6 puede tratarlas como experimento; "
            "como feature aportarán poco al pronóstico."
        ),
        (
            f"Las series difieren {cifra(resumen_dispersion['razon_serie_max_min'], '.0f')}× en nivel y el "
            f"tamaño de la tienda correlaciona {cifra(resumen_tiendas['corr_nivel_m2'], '.2f')} con su "
            "demanda. Un modelo global necesita el nivel de la serie (objetivo escalado o en log) y los "
            "atributos de la tienda."
        ),
        (
            f"Solo {_o['evaluados']:,} de {_o['posibles']:,} orígenes tienen `fut7` completo y señal: el "
            "backtest de E3 debe reportar su cobertura además del error."
        ),
    ]
    mo.md(
        "## 6 · Implicaciones para el modelado\n\n"
        + "\n".join(f"{i}. {t}" for i, t in enumerate(implicaciones, 1))
    )
    return (implicaciones,)


@app.cell
def _(
    UMBRAL_ATIPICO,
    fuente_eda,
    implicaciones,
    json,
    mo,
    resumen_atipicos,
    resumen_calidad,
    resumen_dispersion,
    resumen_efectos,
    resumen_festivos,
    resumen_nivel,
    resumen_promociones,
    resumen_semana,
    resumen_sistema,
    resumen_tendencia,
    resumen_tiendas,
    ruta_salida,
):
    _limpio = (
        "Día limpio: sin Buen Fin, quincena, Navidad, Semana Santa, festivo ni día atípico, y con units_sold."
    )
    definiciones = {
        "fuente.panel_sha256": (
            "SHA-256 del panel leído; si cambia, las cifras de este resumen ya no aplican."
        ),
        "fuente.reporte_calidad_sha256": "SHA-256 del reporte de calidad de E1 que se leyó.",
        "calidad": "Copiado del reporte de E1 (artifacts/calidad/reporte.json), sin recalcular.",
        "demanda.nivel_medio_por_categoria": "Media de units_sold por serie y día, en filas observadas.",
        "demanda.participacion_pct_por_categoria": "Porcentaje de las unidades totales del periodo.",
        "demanda.tendencia.crecimiento_interanual_pct": (
            "Media, en los meses presentes en ambos años, del índice mensual del año siguiente entre el del "
            "año anterior, menos 1. Índice mensual: media en días limpios de units / media de la serie en "
            "ese día de la semana en todos sus días limpios. " + _limpio
        ),
        "demanda.tendencia.mes_mas_alto": "Mes con el índice mensual más alto (definido arriba).",
        "demanda.perfil_dia_semana": (
            "Media en días limpios de units / media de la serie en los días limpios del mismo mes. " + _limpio
        ),
        "demanda.dispersion_series.cv_residual_mediano": (
            "Mediana entre series de la desviación estándar de units / esperado en días limpios (ver "
            "efectos_calendario)."
        ),
        "demanda.dispersion_series.razon_serie_max_min": "Media de la serie más grande / la más chica.",
        "demanda.efectos_calendario": (
            "Multiplicador = media de units / esperado en los días del evento. Esperado: media de la misma "
            "serie, el mismo mes y el mismo día de la semana en días limpios. Se excluyen los días con otro "
            "evento, con un festivo ajeno al evento o atípicos. Fin de semana: media de units / media de "
            "días hábiles limpios de la serie y el mes, en sábado y domingo, entre la misma media en lunes a "
            "viernes. null si el evento no tiene días en el periodo. " + _limpio
        ),
        "demanda.dispersion_efectos_por_categoria": (
            "El mismo multiplicador por categoría; rango_pct = (máximo / mínimo − 1) × 100."
        ),
        "demanda.festivos": (
            "Por fecha, el mismo cociente que los efectos de calendario, para festivos que no son Buen Fin "
            "ni Semana Santa. Si el festivo cae en Navidad o quincena, incluye también ese efecto."
        ),
        "demanda.dias_atipicos": (
            "Índice diario = suma de units_sold observadas / suma de las medianas de esas series. Atípico "
            f"si el índice < {UMBRAL_ATIPICO}. pct_tiendas_bajo_umbral: porcentaje de tiendas cuyo índice "
            "propio queda debajo del umbral ese día."
        ),
        "promociones.tasa_pct": "Porcentaje de filas existentes con has_promotion = 1, por corte.",
        "promociones.quintil_nivel_previo": (
            "Quintiles de la media de units_sold de t−7 a t−1 entre la media de la serie."
        ),
        "promociones.rango_tasa_pp": "Máximo menos mínimo de la tasa entre los valores del corte, en puntos.",
        "promociones.autocorrelacion": (
            "Correlación de has_promotion con su valor 1 y 7 días antes, centrando cada serie en su media."
        ),
        "promociones.lift_ingenuo_pct": (
            "Por serie, media de units_sold con promoción / sin promoción − 1, sin controlar calendario; se "
            "promedia entre las series de la categoría. IC95 normal con la desviación estándar entre series."
        ),
        "tiendas.corr_nivel_m2": (
            "Correlación de Pearson entre los m² y las unidades por día de la tienda (suma de las medias de "
            "sus series)."
        ),
        "tiendas.por_formato_detalle.unidades_por_1000_m2": (
            "Media entre tiendas de unidades por día / m² × 1000."
        ),
        "sistema_actual.wape_pct": (
            "WAPE = Σ|pronóstico − fut7| / Σ fut7 × 100, sobre los orígenes evaluados."
        ),
        "sistema_actual.sesgo_pct": (
            "Sesgo = Σ(pronóstico − fut7) / Σ fut7 × 100; positivo = sobrepronóstico."
        ),
        "sistema_actual.fut7": (
            "Suma de units_sold de T+1 a T+7; el origen se evalúa solo si los 7 días existen."
        ),
        "sistema_actual.senal_T_factor_053": (
            "replenishment_signal fechada en T / 0.53 × 7 (usa ventas de T+1)."
        ),
        "sistema_actual.senal_T_menos_1_factor_ajustado": (
            "replenishment_signal fechada en T−1 / factor del reporte de E1 × 7; no usa información futura."
        ),
        "sistema_actual.ma7": "Media de units_sold de T−6 a T (ignora nulos) × 7.",
        "sistema_actual.wape_por_evento_pct": (
            "WAPE por segmento de la ventana objetivo: buen_fin si contiene un día de Buen Fin; navidad si "
            "no, pero contiene uno de Navidad; resto en otro caso."
        ),
        "implicaciones": "Conclusiones para el modelado; cada una cita una cifra de este resumen.",
    }
    resumen = {
        "fuente": fuente_eda,
        "calidad": resumen_calidad,
        "demanda": {
            **resumen_nivel,
            "tendencia": resumen_tendencia,
            "estacionalidad_semanal": resumen_semana,
            "dispersion_series": resumen_dispersion,
            **resumen_efectos,
            "festivos": resumen_festivos,
            "dias_atipicos": resumen_atipicos,
        },
        "promociones": resumen_promociones,
        "tiendas": resumen_tiendas,
        "sistema_actual": resumen_sistema,
        "implicaciones": implicaciones,
        "definiciones": definiciones,
    }
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str) + "\n")
    mo.md(f"Resumen escrito en `{ruta_salida}`.")
    return


if __name__ == "__main__":
    app.run()
