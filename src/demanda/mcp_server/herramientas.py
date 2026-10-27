"""Las herramientas de solo lectura sobre los artefactos: funciones puras de (directorio, argumentos) a dict.

Cada una lee uno o dos artefactos y devuelve un resumen compacto, sin rutas y sin filas de datos (salvo
`get_forecast`, que devuelve la serie de una tienda para clientes interactivos). Los argumentos se validan
contra el catálogo de los propios artefactos, y si falta un artefacto el error dice qué comando lo genera.
El servidor MCP (`servidor.py`) las expone tal cual; los agentes, las plantillas y la caché las llaman con
los mismos argumentos y reciben lo mismo.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from demanda.seleccion.campeon import campeon_y_retador

# Cada artefacto con el comando que lo escribe
COMANDOS = {
    "calidad/reporte.json": "make calidad",
    "eda/resumen.json": "make eda",
    "backtest/metricas.json": "make backtest",
    "backtest/pronosticos.parquet": "make backtest",
    "modelos/seleccion.json": "make seleccion",
    "drift/monitor.json": "make drift",
    "decision/impacto.json": "make decision",
    "causal/promociones.json": "make causal",
}
# Los segmentos de `metricas.json` que se pueden pedir; `corte` y `horizonte` son diagnósticos
SEGMENTOS_METRICAS = ("global", "evento", "formato", "region", "nse", "categoria", "corte", "horizonte")
# Más alertas que estas por semana se resumen por chequeo en lugar de listarse
MAX_POR = 4


class ArtefactoFaltante(FileNotFoundError):
    """Un artefacto que la herramienta necesita no existe todavía."""


class ArgumentoInvalido(ValueError):
    """Un argumento fuera del catálogo de los artefactos."""


def _ruta(artefactos: Path, relativa: str) -> Path:
    ruta = artefactos / relativa
    if not ruta.exists():
        raise ArtefactoFaltante(f"falta el artefacto {relativa}; genéralo con `{COMANDOS[relativa]}`")
    return ruta


def _leer(artefactos: Path, relativa: str) -> dict[str, Any]:
    return json.loads(_ruta(artefactos, relativa).read_text())


def _validar(valor: Any, catalogo: Sequence[Any], nombre: str) -> None:
    if valor not in catalogo:
        opciones = ", ".join(str(c) for c in catalogo)
        raise ArgumentoInvalido(f"{nombre} inválido: {valor!r}; valores válidos: {opciones}")


def _tomar(d: Mapping[str, Any], *claves: str) -> dict[str, Any]:
    """Las claves de `d` que existen, en el orden dado."""
    return {k: d[k] for k in claves if k in d}


def _estimacion(e: Mapping[str, Any]) -> dict[str, Any]:
    """Lo que se cita de un coeficiente de E6: efecto en %, su IC95 y el valor p."""
    return _tomar(e, "efecto_pct", "ic95_pct", "p_valor", "p_holm", "significativo_holm")


# --- calidad y EDA -------------------------------------------------------------------------------------


def get_data_quality(artefactos: Path) -> dict[str, Any]:
    """Contratos, malla, conteos inconsistentes, nulos y la fuga de `replenishment_signal` (E1)."""
    r = _leer(artefactos, "calidad/reporte.json")
    conteos, fuga = r["conteos"], r["fuga_replenishment_signal"]
    return {
        "contratos": r["contratos"],
        "dimensiones": r["dimensiones"],
        "malla": r["malla"],
        "conteos": {
            **_tomar(conteos, "regla", "filas_evaluables", "inconsistentes", "pct_inconsistentes"),
            "uniformidad_pct": conteos["uniformidad_pct"],
            "avg_ticket": conteos["avg_ticket"],
        },
        "nulos": {
            columna: _tomar(n, "nulos", "pct", "patron", "fechas") for columna, n in r["nulos"].items()
        },
        "fuga_replenishment_signal": _tomar(
            fuga,
            "criterio",
            "mejor_ventana",
            "factor",
            "pesos_por_desfase",
            "corr_con_units_sold",
            "incluye_el_dia",
            "usa_informacion_futura",
            "rezago_minimo_desde_el_origen",
        ),
        "decisiones": r["decisiones"],
    }


def get_eda_summary(artefactos: Path) -> dict[str, Any]:
    """El resumen del EDA (E2): demanda, calendario, promociones, tiendas, sistema actual e implicaciones."""
    r = _leer(artefactos, "eda/resumen.json")
    demanda, promo, tiendas = r["demanda"], r["promociones"], r["tiendas"]
    return {
        "demanda": {
            **_tomar(
                demanda,
                "nivel_medio_por_categoria",
                "nivel_medio_por_formato",
                "participacion_pct_por_categoria",
                "tendencia",
                "estacionalidad_semanal",
                "dispersion_series",
                "efectos_calendario",
                "dispersion_efectos_por_categoria",
                "dias_por_efecto",
                "dias_atipicos",
            ),
            "festivos": [_tomar(f, "fecha", "nombre", "multiplicador") for f in demanda["festivos"]],
        },
        "promociones": _tomar(
            promo,
            "tasa_global_pct",
            "rango_tasa_pp",
            "autocorrelacion",
            "lift_ingenuo_pct",
            "lift_ingenuo_ic95_pct",
            "categorias_con_ic95_sin_cero",
            "nota",
        ),
        "tiendas": _tomar(tiendas, "tiendas", "por_formato", "por_region", "por_nse", "corr_nivel_m2"),
        "sistema_actual": _tomar(
            r["sistema_actual"],
            "protocolo",
            "wape_pct",
            "sesgo_pct",
            "wape_por_evento_pct",
            "aumento_wape_al_quitar_fuga_pp",
            "ventaja_sobre_ma7_sin_fuga_pp",
        ),
        "implicaciones": r["implicaciones"],
    }


# --- pronóstico, decisión y drift ----------------------------------------------------------------------


def _seleccion(s: Mapping[str, Any]) -> dict[str, Any]:
    """La regla de campeón de E4 ya evaluada: criterios, pruebas DM, justificación y sensibilidad."""
    campeon, retador = campeon_y_retador(s)
    regla = s["regla"]
    sens = s["sensibilidad"]
    return {
        "regla": {
            **_tomar(regla, "incumbente", "alfa", "margen_resto_pp", "sesgo_max_pct"),
            "criterios": regla["criterios"],
        },
        "campeon": campeon,
        "retador": retador,
        "mejor_baseline": s["mejor_baseline"],
        "cortes": s["cortes"],
        "modelos": {
            m: {
                **_tomar(
                    v,
                    "wape_semanal_global_pct",
                    "wape_semanal_resto_pct",
                    "sesgo_semanal_max_abs_pct",
                    "sesgo_semanal_peor_subgrupo",
                ),
                "cumple": None
                if v["criterios"] is None
                else {c: x["cumple"] for c, x in v["criterios"].items()},
                "pasa": v["pasa"],
            }
            for m, v in s["modelos"].items()
        },
        "pruebas_dm": [
            _tomar(p, "a", "b", "segmento", "n_cortes", "diferencia_wape_pp", "p_valor", "a_favor_de")
            for p in s["pruebas_dm"]
        ],
        "justificacion": s["justificacion"],
        "sensibilidad": {
            "configuraciones": len(sens),
            "campeones": sorted({f["campeon"] for f in sens}),
            "pasan_alguna_vez": sorted({m for f in sens for m in f["pasan"]}),
        },
    }


def get_metrics(artefactos: Path, segmento: str = "global") -> dict[str, Any]:
    """WAPE, sesgo y cobertura por modelo en cada nivel del segmento (E3), más la regla de campeón (E4)."""
    m = _leer(artefactos, "backtest/metricas.json")
    s = _leer(artefactos, "modelos/seleccion.json")
    disponibles = [g for g in SEGMENTOS_METRICAS if any(f["segmento"] == g for f in m["segmentos"])]
    _validar(segmento, disponibles, "segmento")
    modelos = m["protocolo"]["modelos"]
    niveles: dict[str, dict[str, Any]] = {}
    for f in m["segmentos"]:
        if f["segmento"] != segmento:
            continue
        nivel = niveles.setdefault(f["nivel"], {"modelos": {modelo: {} for modelo in modelos}})
        if f["granularidad"] == "semanal" and f["metrica"] == "wape_pct":
            nivel["pares_semanales"] = {"evaluados": f["n"], "total": f["n_total"]}
        nombre = f"{f['granularidad']}_{f['metrica']}"
        if nombre in ("semanal_wape_pct", "semanal_sesgo_pct", "diaria_wape_pct", "diaria_cobertura_80_pct"):
            nivel["modelos"][f["modelo"]][nombre] = f["valor"]
    definiciones = m["definiciones"]
    return {
        "segmento": segmento,
        "segmentos_disponibles": disponibles,
        "modelos": modelos,
        "cobertura_evaluacion": m["cobertura"],
        "niveles": niveles,
        "definiciones": {
            **_tomar(definiciones["metricas"], "wape_pct", "sesgo_pct", "cobertura_80_pct"),
            "semanal": definiciones["granularidad"]["semanal"],
            "evento": definiciones["segmentos"]["evento"],
        },
        "seleccion": _seleccion(s),
    }


def get_forecast(artefactos: Path, store_id: str, category: str, origen: str | None = None) -> dict[str, Any]:
    """El pronóstico diario del campeón y del retador con su P10–P90 y el real, en un corte del backtest.

    Devuelve la serie de una tienda: es para clientes interactivos, no para los insights que se versionan."""
    pronosticos = _ruta(artefactos, "backtest/pronosticos.parquet")
    campeon, retador = campeon_y_retador(_leer(artefactos, "modelos/seleccion.json"))
    scan = pl.scan_parquet(pronosticos)
    catalogo = scan.select(
        pl.col("store_id").unique().sort().implode(),
        pl.col("category").unique().sort().implode(),
        pl.col("origen").unique().sort().cast(pl.String).implode(),
    ).collect()
    tiendas, categorias, origenes = (catalogo[c][0].to_list() for c in catalogo.columns)
    _validar(store_id, tiendas, "store_id")
    _validar(category, categorias, "category")
    origen = origenes[-1] if origen is None else origen
    _validar(origen, origenes, "origen")
    serie = (
        scan.filter(
            pl.col("store_id") == store_id,
            pl.col("category") == category,
            pl.col("origen").cast(pl.String) == origen,
            pl.col("modelo").is_in([campeon, retador]),
        )
        .sort("modelo", "h")
        .collect()
    )

    def r(x: float | None) -> float | None:
        return None if x is None else round(x, 2)

    dias: dict[str, dict[str, Any]] = {}
    for f in serie.iter_rows(named=True):
        dia = dias.setdefault(str(f["date"]), {"date": str(f["date"]), "h": f["h"], "y": r(f["y"])})
        papel = "campeon" if f["modelo"] == campeon else "retador"
        dia[papel] = {"yhat": r(f["yhat"]), "p10": r(f["p10"]), "p90": r(f["p90"])}
    return {
        "store_id": store_id,
        "category": category,
        "origen": origen,
        "origenes_disponibles": origenes,
        "campeon": campeon,
        "retador": retador,
        "dias": sorted(dias.values(), key=lambda d: d["h"]),
    }


def _chequeo(por: str) -> str:
    """De qué chequeo viene una alerta: «region=Centro · nivel_demanda» → nivel_demanda; las de tienda
    («tienda=STR_055») se cuentan como «tienda», sin el identificador."""
    return por.split(" · ", 1)[1] if " · " in por else por.split("=", 1)[0]


def _recomendacion(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Una recomendación del monitor; si la piden muchas alertas, se cuentan por chequeo."""
    por = rec["por"]
    salida = {"codigo": rec["codigo"], "texto": rec["texto"]}
    if len(por) <= MAX_POR:
        return {**salida, "por": por}
    chequeos = Counter(_chequeo(p) for p in por)
    return {**salida, "alertas": len(por), "por_chequeo": dict(sorted(chequeos.items()))}


def _semana_resumida(s: Mapping[str, Any]) -> dict[str, Any]:
    alertas = [a for a in s["alertas"] if a["severidad"] == "alerta"]
    inesperadas = [a for a in alertas if a["clasificacion"] == "inesperada"]
    return {
        **_tomar(s, "origen", "periodo", "evento"),
        "dia_especial": bool(s["dias_especiales"]),
        "inesperadas": len(inesperadas),
        "esperadas": len(alertas) - len(inesperadas),
        "inesperadas_por_chequeo": dict(sorted(Counter(a["chequeo"] for a in inesperadas).items())),
        "recomendaciones": [_recomendacion(rec) for rec in s["recomendaciones"]],
    }


def get_drift_alerts(artefactos: Path, semana: str | None = None) -> dict[str, Any]:
    """El monitor de drift (E5): resumen por capa, semanas con alertas inesperadas y sus recomendaciones,
    la vigilancia del retador y la validación con drift inyectado. Con `semana` (el origen T de la semana),
    el detalle de esa semana."""
    d = _leer(artefactos, "drift/monitor.json")
    semanas = {s["origen"]: s for s in d["semanas"]}
    if semana is not None:
        _validar(semana, list(semanas), "semana")
        s = semanas[semana]
        return {
            **_tomar(s, "origen", "periodo", "desde", "hasta", "evento", "dias_especiales", "desempeno"),
            "alertas": [
                _tomar(
                    a,
                    "capa",
                    "chequeo",
                    "segmento",
                    "nivel",
                    "modelo",
                    "medida",
                    "valor",
                    "referencia",
                    "umbral",
                    "severidad",
                    "clasificacion",
                )
                for a in s["alertas"]
            ],
            "recomendaciones": [_recomendacion(rec) for rec in s["recomendaciones"]],
        }
    resumen, vig, val, conf = d["resumen"], d["vigilancia_retador"], d["validacion"], d["configuracion"]
    ultimo = vig["por_corte"][-1] if vig["por_corte"] else {}
    return {
        "campeon": d["campeon"],
        "retador": d["retador"],
        "configuracion": {
            **_tomar(conf, "referencia", "segmentos", "esperadas_por_calendario", "banda_wape"),
            "umbrales": _tomar(
                conf["umbrales"], "psi_alerta", "psi_vigilar", "banda_wape", "sesgo_localizado_pp"
            ),
        },
        "resumen": {
            periodo: _tomar(
                resumen[periodo], "semanas", "semanas_con_dia_especial", "por_capa", "recomendaciones"
            )
            for periodo in ("calibracion", "backtest")
        }
        | {"falsas_alarmas_calibracion": resumen["falsas_alarmas_calibracion"]},
        "semanas_con_inesperadas": [
            _semana_resumida(s)
            for s in d["semanas"]
            if any(a["severidad"] == "alerta" and a["clasificacion"] == "inesperada" for a in s["alertas"])
        ],
        "vigilancia_retador": {
            **_tomar(vig, "final", "minima", "coincide_con_seleccion"),
            "umbral_p": ultimo.get("umbral_p"),
            "wape_acumulado_pct_final": ultimo.get("wape_acumulado_pct"),
            "cortes_gana_retador": ultimo.get("cortes_gana_retador"),
            "cortes_cerrados": ultimo.get("cortes_cerrados"),
            "reevaluar_en_algun_corte": any(c["reevaluar"] for c in vig["por_corte"]),
        },
        "validacion": {
            "todos_detectados": val["todos_detectados"],
            "escenarios": [
                _tomar(
                    e,
                    "escenario",
                    "descripcion",
                    "origen",
                    "periodo",
                    "detectado",
                    "recomendacion_esperada",
                    "recomendaciones",
                    "alertas_inesperadas",
                )
                for e in val["escenarios"]
            ],
        },
        "semanas_disponibles": list(semanas),
    }


def get_decision_impact(artefactos: Path) -> dict[str, Any]:
    """La regla de decisión (E5): políticas, impacto global y por evento, pruebas sobre el costo,
    sensibilidad a la fracción crítica y conclusión. Los costos salen de supuestos ilustrativos."""
    d = _leer(artefactos, "decision/impacto.json")
    columnas = (
        "politica",
        "pares",
        "costo",
        "sobreinventario_pct",
        "quiebre_pct",
        "fill_rate_pct",
        "nivel_servicio_pct",
        "fc_objetivo_pct",
        "stock_sobre_demanda",
    )
    por_evento: dict[str, list[dict[str, Any]]] = {}
    for f in d["impacto"]:
        if f["segmento"] == "evento":
            por_evento.setdefault(f["nivel"], []).append(_tomar(f, *columnas))
    return {
        "campeon": d["campeon"],
        "retador": d["retador"],
        "politicas": d["politicas"],
        "supuestos": {
            "nota": "márgenes y costos de mantener ilustrativos; el precio sale de los datos",
            "costos": d["supuestos"]["costos"],
            "fracciones_sensibilidad": d["supuestos"]["fracciones_sensibilidad"],
        },
        "impacto_global": [_tomar(f, *columnas) for f in d["impacto"] if f["segmento"] == "global"],
        "impacto_por_evento": por_evento,
        "pruebas_costo": [
            _tomar(
                p,
                "a",
                "b",
                "segmento",
                "n_cortes",
                "costo_a",
                "costo_b",
                "diferencia_pct",
                "p_valor",
                "a_favor_de",
            )
            for p in d["pruebas_costo"]
        ],
        "sensibilidad": {
            **_tomar(d["sensibilidad"], "orden_base", "orden_se_mantiene"),
            "fracciones": [
                {
                    "fraccion_critica": f["fraccion_critica"],
                    "orden": f["orden"],
                    "costo_global": {p: v["costo"] for p, v in f["global"].items()},
                }
                for f in d["sensibilidad"]["fracciones"]
            ],
        },
        "conclusion": d["conclusion"],
        "definiciones": _tomar(
            d["definiciones"],
            "costo",
            "fill_rate_pct",
            "nivel_servicio_pct",
            "sobreinventario",
            "quiebre",
            "q",
        ),
    }


# --- promociones ---------------------------------------------------------------------------------------


def get_promo_effect(artefactos: Path) -> dict[str, Any]:
    """El experimento de promociones (E6): balance, efecto, canales, heterogeneidad, dinámica, potencia,
    negocio y conclusión."""
    p = _leer(artefactos, "causal/promociones.json")
    bal, pot, neg = p["balance"], p["potencia"], p["negocio"]
    semanal = pot["diseno_semanal"]
    pruebas = {f["prueba"]: f for f in bal["familia"]["pruebas"]}
    return {
        "muestra": _tomar(
            p["muestra"],
            "filas",
            "series",
            "tiendas",
            "desde",
            "hasta",
            "dias_promocion",
            "tasa_promocion_pct",
        ),
        "balance": {
            **_tomar(bal, "se_sostiene", "fallan", "asignacion_se_sostiene", "pruebas_de_medicion"),
            "b1": _tomar(bal["b1"], "descripcion", "p_valor", "cumple"),
            "b3": _tomar(bal["b3"], "descripcion", "p_valor", "cumple"),
            "familia": _tomar(bal["familia"], "n_pruebas", "alfa", "p_holm_min", "rechazadas"),
            "que_fallan": {
                nombre: _tomar(pruebas[nombre], "p_holm", "tasa_con_promocion_pct", "tasa_sin_promocion_pct")
                for nombre in bal["fallan"]
                if nombre in pruebas
            },
        },
        "efecto_principal": _tomar(
            p["efecto_principal"],
            "modelo",
            "efecto_pct",
            "ic95_pct",
            "p_valor",
            "p_aleatorizacion",
            "inferencia_principal",
            "p_valor_principal",
            "significativo",
        ),
        "canales": {
            nombre: {**_tomar(c, "resultado", "secundario"), **_estimacion(c), **_tomar(c, "advertencia")}
            for nombre, c in p["canales"].items()
        },
        "robustez": {
            nombre: {**_tomar(r, "descripcion"), **_estimacion(r)}
            for nombre, r in p["robustez"].items()
            if "efecto_pct" in r
        },
        "calibracion_aa": _tomar(
            p["calibracion_aa"], "tasa_pct", "intervalo_99_pct", "dentro", "inferencia_principal"
        ),
        "heterogeneidad": {
            **{
                dimension: {
                    "grupos": {g: _estimacion(v) for g, v in h["grupos"].items()},
                    "igualdad": _tomar(h["igualdad"], "p_valor", "hay_heterogeneidad"),
                }
                for dimension, h in p["heterogeneidad"].items()
                if dimension != "familia_holm"
            },
            "significativos_holm": p["heterogeneidad"]["familia_holm"]["significativos"],
        },
        "dinamica": {
            "suma_rezagos": _estimacion(p["dinamica"]["suma_rezagos"]),
            "neto": _estimacion(p["dinamica"]["neto"]),
            "adelanto": p["dinamica"]["adelanto"],
        },
        "cruzado": {
            "otras_categorias": _estimacion(p["cruzado"]["otras_categorias"]),
            "interpretacion": p["cruzado"]["interpretacion"],
        },
        "potencia": {
            "alfa": pot["diseno_actual"]["alfa"],
            "potencia": pot["diseno_actual"]["potencia"],
            "mde_diario_pct": pot["diseno_actual"]["global"]["mde_pct"],
            "mde_diario_por_categoria_pct": {
                g: v["mde_pct"] for g, v in pot["diseno_actual"]["categoria"].items()
            },
            "semanal": {
                **_tomar(semanal, "descripcion", "semanas"),
                "mde_actual_pct": semanal["mde_actual"]["mde_pct"],
                "mde_por_semanas_pct": {str(f["semanas"]): f["mde_pct"] for f in semanal["por_semanas"]},
                "semanas_necesarias": semanal["semanas_necesarias"],
            },
        },
        "negocio": {
            "descuento_implicito": neg["descuento_implicito"],
            "por_categoria": {
                g: {
                    **_tomar(
                        c,
                        "margen_bruto",
                        "precio_lista",
                        "lift_pct",
                        "lift_ic95_pct",
                        "incrementales_por_dia_promocion",
                        "valor_maximo_sin_descuento_por_dia",
                    ),
                    "por_descuento": [
                        _tomar(f, "descuento", "origen", "lift_equilibrio_pct", "veredicto")
                        for f in c["por_descuento"]
                    ],
                }
                for g, c in neg["por_categoria"].items()
            },
            "por_descuento": neg["por_descuento"],
        },
        "supuestos": {
            "negocio": p["supuestos"]["negocio"],
            **_tomar(p["supuestos"], "estimacion", "no_anticipacion", "sin_contaminacion"),
        },
        "conclusion": p["conclusion"],
    }


HERRAMIENTAS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_data_quality": get_data_quality,
    "get_eda_summary": get_eda_summary,
    "get_metrics": get_metrics,
    "get_forecast": get_forecast,
    "get_drift_alerts": get_drift_alerts,
    "get_decision_impact": get_decision_impact,
    "get_promo_effect": get_promo_effect,
}
# Los artefactos que lee cada herramienta: su sha256 va a la caché por trazabilidad
FUENTES: dict[str, tuple[str, ...]] = {
    "get_data_quality": ("calidad/reporte.json",),
    "get_eda_summary": ("eda/resumen.json",),
    "get_metrics": ("backtest/metricas.json", "modelos/seleccion.json"),
    "get_forecast": ("backtest/pronosticos.parquet", "modelos/seleccion.json"),
    "get_drift_alerts": ("drift/monitor.json",),
    "get_decision_impact": ("decision/impacto.json",),
    "get_promo_effect": ("causal/promociones.json",),
}


def llamar(artefactos: Path, herramienta: str, argumentos: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Llama una herramienta por su nombre, como lo haría un cliente MCP."""
    _validar(herramienta, list(HERRAMIENTAS), "herramienta")
    return HERRAMIENTAS[herramienta](artefactos, **(argumentos or {}))
