"""Las conclusiones sin LLM: se arman en Python con los campos de las respuestas de las herramientas.

Son deterministas y pasan por el mismo validador que los agentes, como autoprueba: toda cifra sale de un
campo de la herramienta, con el formato con el que se cita. Son correctas pero planas: no priorizan ni
conectan hallazgos entre dominios; el orquestador de plantilla solo toma el primer insight de cada uno.
"""

from collections.abc import Callable, Mapping
from typing import Any

from demanda.agentes.modelos import DOMINIOS, Evidencia, Herramienta, Insight

Respuestas = Mapping[str, Mapping[str, Any]]


def _ev(cifra: str, que_mide: str, herramienta: Herramienta) -> Evidencia:
    return Evidencia(cifra=cifra, que_mide=que_mide, herramienta=herramienta)


def _menos_mas(x: float) -> str:
    return "menos" if x < 0 else "más"


def _p(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


# --- datos ---------------------------------------------------------------------------------------------


def _fuga(q: Mapping[str, Any], e: Mapping[str, Any]) -> Insight:
    fuga = q["fuga_replenishment_signal"]
    ventana, wape = fuga["mejor_ventana"], e["sistema_actual"]["wape_pct"]
    con_fuga, sin_fuga = wape["senal_T_factor_ajustado"], wape["senal_T_menos_1_factor_ajustado"]
    futuro = "mira al futuro" if fuga["usa_informacion_futura"] else "no mira al futuro"
    return Insight(
        titulo="La señal de reposición es fuga de información",
        hallazgo=(
            "replenishment_signal fechada en t equivale a la media de units_sold de "
            f"t{ventana['desde']:+d} a t{ventana['hasta']:+d} (correlación {ventana['corr']:.4f}), así que "
            f"{futuro}. Con la señal del mismo día el sistema actual tiene un WAPE de {con_fuga:.2f} %; sin "
            f"la fuga, de {sin_fuga:.2f} %."
        ),
        evidencia=[
            _ev(f"{ventana['corr']:.4f}", "correlación de la señal con la media móvil", "get_data_quality"),
            _ev(f"{con_fuga:.2f} %", "WAPE del sistema actual con la señal de T", "get_eda_summary"),
            _ev(f"{sin_fuga:.2f} %", "WAPE del sistema actual con la señal de T−1", "get_eda_summary"),
        ],
        accion=(
            "Usar la señal solo rezagada (con origen T, fechada en "
            f"T−{fuga['rezago_minimo_desde_el_origen']} o antes) y comparar cualquier modelo contra el "
            "sistema sin fuga."
        ),
        confianza="alta",
    )


def _conteos(q: Mapping[str, Any]) -> Insight:
    pct = q["conteos"]["pct_inconsistentes"]
    ticket = q["conteos"]["avg_ticket"]["sobreestimacion_media_pct_en_inconsistentes"]
    return Insight(
        titulo="Conteos de transacciones inconsistentes",
        hallazgo=(
            f"En el {pct:.2f} % de las filas, cash_transactions + card_transactions es menor que "
            f"total_transactions, y ahí avg_ticket sobreestima el ticket en {ticket:.2f} % en promedio. El "
            "experimento de promociones mostró que esas filas son los días con promoción."
        ),
        evidencia=[
            _ev(f"{pct:.2f} %", "filas con conteos inconsistentes", "get_data_quality"),
            _ev(f"{ticket:.2f} %", "sobreestimación media de avg_ticket en esas filas", "get_data_quality"),
        ],
        accion="No usar avg_ticket ni total_transactions como medida de tráfico sin corregir el registro.",
        confianza="alta",
    )


def _huecos(q: Mapping[str, Any]) -> Insight:
    malla, nulos = q["malla"], q["nulos"]["units_sold"]["pct"]
    return Insight(
        titulo="Huecos y nulos que no se imputan",
        hallazgo=(
            f"Faltan {malla['faltantes']} de las {malla['esperadas']:,} filas de la malla tienda × "
            f"categoría × día, y units_sold es nulo en el {nulos:.2f} % de las filas, de forma dispersa."
        ),
        evidencia=[
            _ev(f"{malla['faltantes']}", "filas faltantes de la malla", "get_data_quality"),
            _ev(f"{nulos:.2f} %", "nulos de units_sold", "get_data_quality"),
        ],
        accion="Evaluar solo con días observados y reportar la cobertura de cada métrica, sin imputar.",
        confianza="alta",
    )


def _calendario(e: Mapping[str, Any]) -> Insight | None:
    efectos, demanda = e["demanda"]["efectos_calendario"], e["demanda"]
    buen_fin, fin_de_semana = efectos.get("buen_fin"), efectos.get("fin_de_semana")
    if buen_fin is None or fin_de_semana is None:
        return None
    razon = demanda["dispersion_series"]["razon_serie_max_min"]
    crecimiento = demanda["tendencia"]["crecimiento_interanual_pct"]
    return Insight(
        titulo="El calendario y la escala de cada serie mandan",
        hallazgo=(
            f"El Buen Fin multiplica la demanda por {buen_fin:.2f} y el fin de semana por "
            f"{fin_de_semana:.2f}; "
            f"las series difieren {razon:.1f} veces en nivel y el crecimiento interanual es de "
            f"{crecimiento:.2f} %."
        ),
        evidencia=[
            _ev(f"{buen_fin:.2f}", "multiplicador del Buen Fin", "get_eda_summary"),
            _ev(f"{fin_de_semana:.2f}", "multiplicador del fin de semana", "get_eda_summary"),
            _ev(f"{razon:.1f}", "razón entre la serie más grande y la más chica", "get_eda_summary"),
        ],
        accion=(
            "Pronosticar con un modelo global que use el calendario y el nivel de cada serie; el Buen Fin "
            "ocurre una vez y su efecto no se puede validar fuera de muestra."
        ),
        confianza="media",
    )


def datos(r: Respuestas) -> list[Insight]:
    q, e = r["get_data_quality"], r["get_eda_summary"]
    insights = [_fuga(q, e), _conteos(q), _huecos(q), _calendario(e)]
    return [i for i in insights if i is not None]


# --- pronóstico ----------------------------------------------------------------------------------------


def _campeon(m: Mapping[str, Any], d: Mapping[str, Any]) -> Insight:
    s = m["seleccion"]
    c, ret = s["campeon"], s["retador"]
    mc, mr = s["modelos"][c], s["modelos"][ret]
    dm = next((p for p in s["pruebas_dm"] if (p["a"], p["b"], p["segmento"]) == (ret, c, "global")), None)
    necesarios = (d["vigilancia_retador"].get("final") or {}).get("cortes_necesarios")
    evidencia = [
        _ev(f"{mr['wape_semanal_global_pct']:.2f} %", f"WAPE semanal global de {ret}", "get_metrics"),
        _ev(f"{mc['wape_semanal_global_pct']:.2f} %", f"WAPE semanal global de {c}", "get_metrics"),
    ]
    if dm is not None and dm["p_valor"] is not None:
        evidencia.append(_ev(_p(dm["p_valor"]), f"valor p de la Diebold-Mariano de {ret}", "get_metrics"))
    plazo = f"; la vigilancia estima {necesarios} cortes para decidir." if necesarios else "."
    return Insight(
        titulo=f"{c} sigue como campeón",
        hallazgo=s["justificacion"],
        evidencia=evidencia,
        accion=f"Mantener {c} y vigilar a {ret} como retador{plazo}",
        confianza="alta",
    )


def _semanas_normales(m: Mapping[str, Any]) -> Insight:
    s = m["seleccion"]
    c, ret = s["campeon"], s["retador"]
    mc, mr = s["modelos"][c], s["modelos"][ret]
    en_eventos = (
        mr["wape_semanal_global_pct"] < mc["wape_semanal_global_pct"]
        and mr["wape_semanal_resto_pct"] >= mc["wape_semanal_resto_pct"]
    )
    cierre = "; la ventaja del retador en el global sale de los eventos." if en_eventos else "."
    return Insight(
        titulo="Las semanas normales y los eventos cuentan historias distintas",
        hallazgo=(
            f"En las semanas normales el WAPE semanal de {c} es {mc['wape_semanal_resto_pct']:.2f} % y el "
            f"de {ret} {mr['wape_semanal_resto_pct']:.2f} %{cierre}"
        ),
        evidencia=[
            _ev(
                f"{mc['wape_semanal_resto_pct']:.2f} %",
                f"WAPE semanal de {c} en semanas normales",
                "get_metrics",
            ),
            _ev(
                f"{mr['wape_semanal_resto_pct']:.2f} %",
                f"WAPE semanal de {ret} en semanas normales",
                "get_metrics",
            ),
        ],
        accion=(
            "Planear los eventos (Buen Fin, fin de año) con el área comercial en lugar de esperar que el "
            "modelo los resuelva."
        ),
        confianza="media",
    )


def _regla(i: Mapping[str, Any]) -> Insight | None:
    globales = [p for p in i["pruebas_costo"] if p["segmento"] == "global"]
    regla = next((p for p in globales if p["b"].startswith("puntual")), None)
    modelo = next((p for p in globales if not p["b"].startswith("puntual")), None)
    if regla is None or modelo is None or None in (regla["diferencia_pct"], modelo["diferencia_pct"]):
        return None
    dr, dm = regla["diferencia_pct"], modelo["diferencia_pct"]
    return Insight(
        titulo="La regla de reposición vale más que cambiar de modelo",
        hallazgo=(
            f"Con los costos supuestos, {regla['a']} cuesta {abs(dr):.1f} % {_menos_mas(dr)} que "
            f"{regla['b']} (p = {_p(regla['p_valor'])}), y {modelo['a']} cuesta {abs(dm):.1f} % "
            f"{_menos_mas(dm)} que {modelo['b']} (p = {_p(modelo['p_valor'])})."
        ),
        evidencia=[
            _ev(f"{dr:.1f} %", "cambio de costo por la regla newsvendor", "get_decision_impact"),
            _ev(f"{dm:.1f} %", "cambio de costo por usar el retador", "get_decision_impact"),
        ],
        accion=(
            "Adoptar la regla newsvendor sobre el campeón; antes de fijar montos, reemplazar los márgenes y "
            "costos de mantener, que son supuestos ilustrativos, por los reales."
        ),
        confianza="media",
    )


def _monitor(d: Mapping[str, Any]) -> Insight:
    detecto = "detectó todos" if d["validacion"]["todos_detectados"] else "no detectó todos"
    inesperadas = d["resumen"]["backtest"]["por_capa"]["entradas"]["inesperadas"]
    tiendas = sorted(
        {
            p.removeprefix("tienda=")
            for semana in d["semanas_con_inesperadas"]
            if semana["periodo"] == "backtest"
            for rec in semana["recomendaciones"]
            for p in rec.get("por", [])
            if p.startswith("tienda=")
        }
    )
    captura = (
        f" La alerta de captura de {', '.join(tiendas)} sí es una falla real de datos." if tiendas else ""
    )
    return Insight(
        titulo="El monitor detecta fallas, pero confunde la estacionalidad con drift",
        hallazgo=(
            f"Con drift inyectado, el monitor {detecto} los escenarios. En el backtest dio {inesperadas} "
            "alertas inesperadas de entradas: las de nivel de noviembre y enero vienen de la estacionalidad "
            f"anual, que su chequeo de nivel confunde con drift.{captura}"
        ),
        evidencia=[
            _ev(f"{inesperadas}", "alertas inesperadas de entradas en el backtest", "get_drift_alerts")
        ],
        accion=(
            "Atender las alertas de captura y no reentrenar por las alertas de nivel de noviembre y enero; "
            "comparar el nivel contra el mismo periodo del año anterior cuando haya historia."
        ),
        confianza="media",
    )


def pronostico(r: Respuestas) -> list[Insight]:
    m, d, i = r["get_metrics"], r["get_drift_alerts"], r["get_decision_impact"]
    insights = [_campeon(m, d), _semanas_normales(m), _regla(i), _monitor(d)]
    return [x for x in insights if x is not None]


# --- promociones ---------------------------------------------------------------------------------------


def _efecto(p: Mapping[str, Any]) -> Insight:
    e, mde = p["efecto_principal"], p["potencia"]["mde_diario_pct"]
    ic = e["ic95_pct"]
    significativo = "significativo" if e["significativo"] else "no significativo"
    return Insight(
        titulo="Las promociones no mueven las unidades en el agregado",
        hallazgo=(
            f"La promoción cambia las unidades en {e['efecto_pct']:+.2f} % (IC95 de {ic[0]:+.2f} a "
            f"{ic[1]:+.2f} %; p = {_p(e['p_valor_principal'])}), {significativo}. El efecto mínimo "
            "detectable "
            f"con el diseño actual es de {mde:.2f} %."
        ),
        evidencia=[
            _ev(f"{e['efecto_pct']:+.2f} %", "efecto de la promoción sobre las unidades", "get_promo_effect"),
            _ev(_p(e["p_valor_principal"]), "valor p principal", "get_promo_effect"),
            _ev(f"{mde:.2f} %", "efecto mínimo detectable diario", "get_promo_effect"),
        ],
        accion="No ampliar las promociones esperando más volumen en el agregado.",
        confianza="alta",
    )


def _excepcion(p: Mapping[str, Any]) -> Insight | None:
    significativos = p["heterogeneidad"]["significativos_holm"]
    if not significativos:
        return None
    dimension, grupo = significativos[0].split(":", 1)
    g = p["heterogeneidad"][dimension]["grupos"][grupo]
    ic = g["ic95_pct"]
    return Insight(
        titulo=f"{grupo} es la excepción",
        hallazgo=(
            f"Con la corrección de Holm, {grupo} tiene un efecto distinto de cero: {g['efecto_pct']:+.2f} % "
            f"(IC95 de {ic[0]:+.2f} a {ic[1]:+.2f} %; p de Holm {g['p_holm']:.4f})."
        ),
        evidencia=[
            _ev(f"{g['efecto_pct']:+.2f} %", f"efecto en {grupo}", "get_promo_effect"),
            _ev(f"{g['p_holm']:.4f}", "valor p ajustado por Holm", "get_promo_effect"),
        ],
        accion=f"Concentrar las pruebas de promoción en {grupo} y medir su retorno con el descuento real.",
        confianza="media",
    )


def _retorno(p: Mapping[str, Any]) -> Insight | None:
    rejilla = [f for f in p["negocio"]["por_descuento"] if f["origen"] == "rejilla"]
    if not rejilla:
        return None
    nadie = all(not f["se_paga"] for f in rejilla)
    menor, mayor = 100 * rejilla[0]["descuento"], 100 * rejilla[-1]["descuento"]
    alcance = "ninguna categoría alcanza" if nadie else "solo algunas categorías alcanzan"
    return Insight(
        titulo="Con los descuentos supuestos, las promociones no se pagan"
        if nadie
        else "Pocas promociones se pagan",
        hallazgo=(
            f"Con los márgenes supuestos y descuentos de {menor:.0f} % a {mayor:.0f} %, {alcance} el lift de "
            "equilibrio. El retorno depende del descuento, que es un supuesto."
        ),
        evidencia=[
            _ev(f"{menor:.0f} %", "menor descuento de la rejilla", "get_promo_effect"),
            _ev(f"{mayor:.0f} %", "mayor descuento de la rejilla", "get_promo_effect"),
        ],
        accion="Revisar el descuento real de cada promoción antes de repetirla: sin él no se sabe si paga.",
        confianza="media",
    )


def _medicion(p: Mapping[str, Any]) -> Insight | None:
    bal, canales = p["balance"], p["canales"]
    if bal["se_sostiene"] or not {"transacciones", "transacciones_efectivo_tarjeta"} <= set(canales):
        return None
    tr, et = canales["transacciones"]["efecto_pct"], canales["transacciones_efectivo_tarjeta"]["efecto_pct"]
    asignacion = "; la asignación sí se sostiene" if bal["asignacion_se_sostiene"] else ""
    return Insight(
        titulo="El tráfico medido con total_transactions está contaminado",
        hallazgo=(
            f"El balance falla en {', '.join(bal['fallan'])}, una prueba del registro y no de la "
            f"asignación{asignacion}. total_transactions sube {tr:.2f} % con promoción, pero contado con "
            f"efectivo más tarjeta cambia {et:+.2f} %."
        ),
        evidencia=[
            _ev(f"{tr:.2f} %", "efecto sobre total_transactions", "get_promo_effect"),
            _ev(f"{et:+.2f} %", "efecto sobre efectivo más tarjeta", "get_promo_effect"),
        ],
        accion="Corregir el registro de total_transactions en los días con promoción antes de usarlo.",
        confianza="alta",
    )


def causal(r: Respuestas) -> list[Insight]:
    p = r["get_promo_effect"]
    insights = [_efecto(p), _excepcion(p), _retorno(p), _medicion(p)]
    return [i for i in insights if i is not None]


PLANTILLAS: dict[str, Callable[[Respuestas], list[Insight]]] = {
    "datos": datos,
    "pronostico": pronostico,
    "causal": causal,
}
assert set(PLANTILLAS) == set(DOMINIOS)


def orquestador(por_dominio: Mapping[str, list[Insight]]) -> list[Insight]:
    """El resumen ejecutivo de plantilla: el primer insight de cada dominio, con el dominio en el título."""
    nombres = {"datos": "Datos", "pronostico": "Pronóstico", "causal": "Promociones"}
    return [
        insights[0].model_copy(update={"titulo": f"{nombres[d]}: {insights[0].titulo}"})
        for d, insights in por_dominio.items()
        if insights
    ]
