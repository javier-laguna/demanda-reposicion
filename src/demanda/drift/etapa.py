"""La etapa del monitor: los chequeos semana a semana, la vigilancia del retador, la validación con drift
inyectado y `monitor.json`.

- Semanas de calibración: los jueves de la rejilla del backtest con 56 días de historia, antes del primer
  corte. En ellas se mide la tasa de falsas alarmas, y las que tienen pronóstico (las de
  `calibracion.parquet`) fijan la banda de WAPE.
- Semanas del backtest: los cortes de `metricas.json`. Es el periodo vigilado.

Solo lleva identificadores de tienda y agregados, no filas ni series, y dos corridas lo escriben igual.
"""

import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from demanda.backtest import calibracion
from demanda.backtest.etapa import verificar_malla
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256
from demanda.drift import inyeccion
from demanda.drift.monitor import (
    CHEQUEOS,
    Revision,
    Umbrales,
    primera_fecha,
    recomendar,
    revisar,
    vigilar_retador,
)
from demanda.seleccion.campeon import campeon_y_retador, verificar_fuente

SEMANAS_VALIDACION = (date(2023, 10, 5), date(2024, 2, 8))
MAGNITUDES = (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30)
DIAS_FALLA = (1, 2, 3)
POTENCIA = {"region": "region", "categoria": "category"}
BONFERRONI = {
    "tienda_sin_datos": "alfa_calidad",
    "tasas_globales": "alfa_calidad",
    "plan_promocion": "alfa_promocion",
}
CAMPOS = (
    "capa",
    "chequeo",
    "segmento",
    "nivel",
    "modelo",
    "medida",
    "valor",
    "referencia",
    "umbral",
    "p_valor",
    "severidad",
    "clasificacion",
    "cuando",
)


class Semana(NamedTuple):
    origen: date
    periodo: str  # calibracion o backtest


def _r(x: float | None, digitos: int = 4) -> float | None:
    return None if x is None else round(float(x), digitos)


def _sig(x: float | None, cifras: int = 4) -> float | None:
    """Con cifras significativas, para los valores p y los umbrales chicos."""
    return None if x is None else float(f"{x:.{cifras}g}")


def semanas(panel: pl.DataFrame, cortes: Sequence[date], u: Umbrales) -> list[Semana]:
    """Las semanas de calibración (los jueves de la rejilla con `dias_referencia` días de historia, antes del
    primer corte) y las del backtest."""
    primera = primera_fecha(panel)
    calibracion_ = []
    origen = cortes[0] - timedelta(days=u.horizonte)
    while origen - timedelta(days=u.dias_referencia - 1) >= primera:
        calibracion_.insert(0, origen)
        origen -= timedelta(days=u.horizonte)
    return [Semana(o, "calibracion") for o in calibracion_] + [Semana(o, "backtest") for o in cortes]


def bandas(revisiones: Sequence[Revision], u: Umbrales) -> dict[str, dict[str, Any]]:
    """Por modelo, la mediana del WAPE semanal en las semanas con pronóstico y la banda, `banda_wape` × esa
    mediana."""
    por_modelo: dict[str, list[float]] = {}
    for r in revisiones:
        for modelo, d in (r.desempeno or {}).items():
            if d["wape_pct"] is not None:
                por_modelo.setdefault(modelo, []).append(d["wape_pct"])
    return {
        m: {
            "semanas": len(w),
            "mediana_pct": statistics.median(w),
            "banda_pct": u.banda_wape * statistics.median(w),
        }
        for m, w in por_modelo.items()
    }


def bonferroni(
    revisiones: Sequence[Revision], vigilancia: Sequence[Mapping[str, Any]], u: Umbrales
) -> dict[str, dict[str, Any]]:
    """Por chequeo con Bonferroni, el α, el umbral de p que se aplicó y cuántas comparaciones corrige. Sale de
    las medidas y de la vigilancia, así describe lo que se aplicó y no un cálculo aparte."""
    salida = {}
    for chequeo, alfa in BONFERRONI.items():
        aplicados = {m["umbral"] for r in revisiones for m in r.medidas if m["chequeo"] == chequeo}
        if len(aplicados) != 1:
            raise ValueError(f"el umbral de {chequeo} no es el mismo en todas las semanas")
        umbral = aplicados.pop()
        salida[chequeo] = {"alfa": getattr(u, alfa), "comparaciones": round(getattr(u, alfa) / umbral)}
        salida[chequeo]["umbral_p"] = _sig(umbral)
    if vigilancia:
        umbral = vigilancia[-1]["umbral_p"]
        salida["vigilancia_retador"] = {
            "alfa": u.alfa_retador,
            "comparaciones": round(u.alfa_retador / umbral),
            "umbral_p": _sig(umbral),
        }
    return salida


def _alerta_json(a: Mapping[str, Any]) -> dict[str, Any]:
    redondeo = {"valor": _r, "referencia": _r, "umbral": _sig, "p_valor": _sig}
    return {c: redondeo[c](a[c]) if c in redondeo else a[c] for c in CAMPOS}


def _desempeno_json(desempeno: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any] | None:
    if desempeno is None:
        return None
    return {
        m: {"wape_pct": _r(d["wape_pct"]), "sesgo_pct": _r(d["sesgo_pct"]), "pares": d["pares"]}
        for m, d in desempeno.items()
    }


def _vigilancia_json(filas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **f,
            "wape_acumulado_pct": {m: _r(w) for m, w in f["wape_acumulado_pct"].items()},
            "dm": None
            if f["dm"] is None
            else {
                "media_d": _r(f["dm"]["media_d"], 2),
                "estadistico": _r(f["dm"]["estadistico"]),
                "p_valor": round(f["dm"]["p_valor"], 6),
                "a_favor_de": f["dm"]["a_favor_de"],
            },
            "umbral_p": _sig(f["umbral_p"]),
        }
        for f in filas
    ]


def _coincide(
    final: Mapping[str, Any] | None, seleccion: Mapping[str, Any], campeon: str, retador: str, n: int
) -> bool | None:
    """Si la DM con todos los cortes es la de `seleccion.json`. None si alguna de las dos no existe."""
    prueba = next(
        (
            p
            for p in seleccion.get("pruebas_dm", [])
            if (p["a"], p["b"], p["segmento"]) == (retador, campeon, "global")
        ),
        None,
    )
    if final is None or prueba is None or prueba["p_valor"] is None:
        return None
    return (
        prueba["n_cortes"] == n
        and round(final["p_valor"], 6) == prueba["p_valor"]
        and round(final["estadistico"], 4) == prueba["estadistico"]
    )


def _resumen(semanas_json: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Por periodo y chequeo: semanas evaluadas, alertas (inesperadas y esperadas), «vigilar» y semanas con
    alguna inesperada. En las semanas de calibración, las inesperadas son las falsas alarmas."""
    resumen: dict[str, Any] = {}
    for periodo in ("calibracion", "backtest"):
        del_periodo = [s for s in semanas_json if s["periodo"] == periodo]
        por_chequeo: dict[str, dict[str, Any]] = {}
        for chequeo, (capa, _, _) in CHEQUEOS.items():
            if chequeo == "calendario":
                continue
            evaluadas = [s for s in del_periodo if chequeo in s["chequeos_corridos"]]
            de_chequeo = [[a for a in s["alertas"] if a["chequeo"] == chequeo] for s in evaluadas]
            alertas = [a for semana in de_chequeo for a in semana if a["severidad"] == "alerta"]
            por_chequeo[chequeo] = {
                "capa": capa,
                "semanas_evaluadas": len(evaluadas),
                "alertas": len(alertas),
                "inesperadas": sum(a["clasificacion"] == "inesperada" for a in alertas),
                "esperadas": sum(a["clasificacion"] == "esperada" for a in alertas),
                "vigilar": sum(a["severidad"] == "vigilar" for semana in de_chequeo for a in semana),
                "semanas_con_inesperada": sum(
                    any(a["severidad"] == "alerta" and a["clasificacion"] == "inesperada" for a in semana)
                    for semana in de_chequeo
                ),
            }
        recomendaciones = Counter(r["codigo"] for s in del_periodo for r in s["recomendaciones"])
        por_capa = {
            capa: {
                clave: sum(v[clave] for v in por_chequeo.values() if v["capa"] == capa)
                for clave in ("alertas", "inesperadas", "esperadas", "vigilar")
            }
            for capa in dict.fromkeys(capa for capa, _, _ in CHEQUEOS.values())
        }
        resumen[periodo] = {
            "semanas": len(del_periodo),
            "semanas_con_dia_especial": sum(bool(s["dias_especiales"]) for s in del_periodo),
            "por_capa": por_capa,
            "por_chequeo": por_chequeo,
            "recomendaciones": dict(sorted(recomendaciones.items())),
        }
    resumen["falsas_alarmas_calibracion"] = {
        c: v["inesperadas"] for c, v in resumen["calibracion"]["por_chequeo"].items()
    }

    def contar(s: Mapping[str, Any], severidad: str, clasificacion: str | None = None) -> int:
        return sum(
            a["severidad"] == severidad and clasificacion in (None, a["clasificacion"]) for a in s["alertas"]
        )

    resumen["por_semana"] = [
        {
            "origen": s["origen"],
            "periodo": s["periodo"],
            "evento": s["evento"],
            "dia_especial": bool(s["dias_especiales"]),
            "inesperadas": contar(s, "alerta", "inesperada"),
            "esperadas": contar(s, "alerta", "esperada"),
            "vigilar": contar(s, "vigilar"),
            "recomendaciones": [r["codigo"] for r in s["recomendaciones"]],
        }
        for s in semanas_json
    ]
    return resumen


def _en_segmento(medidas: Sequence[Mapping[str, Any]], segmento: str, nivel: str) -> list[Mapping[str, Any]]:
    return [m for m in medidas if m["segmento"] == segmento and m["nivel"] == nivel]


def _detectado_por(alertas: Sequence[Mapping[str, Any]], segmento: str, nivel: str) -> list[str]:
    """Los chequeos con alerta inesperada en el segmento, con el modelo en los de desempeño."""
    return sorted(
        {
            a["chequeo"] + (f":{a['modelo']}" if a["modelo"] else "")
            for a in _en_segmento(alertas, segmento, nivel)
            if a["severidad"] == "alerta" and a["clasificacion"] == "inesperada"
        }
    )


def validar(
    panel: pl.DataFrame,
    pronosticos: pl.DataFrame,
    semanas_val: Sequence[Semana],
    banda: Mapping[str, float],
    modelos: Sequence[str],
    u: Umbrales,
) -> dict[str, Any]:
    """Los tres escenarios en cada semana de validación y la potencia: saltos de nivel por región y por
    categoría, y fallas de 1 a 3 días."""
    escenarios = []
    potencia_salto, potencia_falla = [], []
    for semana in semanas_val:
        banda_semana = banda if semana.periodo == "backtest" else None

        def correr(
            p: pl.DataFrame, s: Semana = semana, b: Mapping[str, float] | None = banda_semana
        ) -> Revision:
            return revisar(p, pronosticos, s.origen, u, b)

        for nombre, esc in inyeccion.ESCENARIOS.items():
            rev = correr(esc.inyectar(panel, semana.origen, u.horizonte))
            alertas = rev.alertas
            codigos = [r["codigo"] for r in recomendar(alertas, rev.sin_precedente)]
            esperadas = [
                {
                    "chequeo": e.chequeo,
                    "segmento": e.segmento,
                    "nivel": e.nivel,
                    "modelo": m,
                    "detectada": hay,
                }
                for e in esc.esperadas(panel)
                for m, hay in inyeccion.encontradas(alertas, e, modelos).items()
            ]
            escenarios.append(
                {
                    "escenario": nombre,
                    "descripcion": esc.descripcion,
                    "origen": str(semana.origen),
                    "periodo": semana.periodo,
                    "esperadas": esperadas,
                    "recomendacion_esperada": esc.recomendacion,
                    "recomendaciones": codigos,
                    "detectado": all(e["detectada"] for e in esperadas) and esc.recomendacion in codigos,
                    "alertas_inesperadas": sum(
                        a["severidad"] == "alerta" and a["clasificacion"] == "inesperada" for a in alertas
                    ),
                }
            )

        for dimension, columna in POTENCIA.items():
            for nivel in sorted(panel[columna].unique().to_list()):
                for cambio in MAGNITUDES:
                    rev = correr(
                        inyeccion.salto_nivel(panel, semana.origen, u.horizonte, 1 + cambio, columna, nivel)
                    )
                    medidas = _en_segmento(rev.medidas, dimension, nivel)
                    psi = next(m["valor"] for m in medidas if m["chequeo"] == "nivel_demanda")
                    sesgos = {
                        m["modelo"]: _r(m["valor"]) for m in medidas if m["chequeo"] == "sesgo_localizado"
                    }
                    potencia_salto.append(
                        {
                            "origen": str(semana.origen),
                            "segmento": dimension,
                            "nivel": nivel,
                            "cambio_pct": round(100 * cambio),
                            "psi_nivel": _r(psi),
                            "sesgo_localizado_pp": sesgos,
                            "detectado_por": _detectado_por(rev.alertas, dimension, nivel),
                        }
                    )
        tienda = inyeccion.primera_tienda(panel)
        for dias in DIAS_FALLA:
            rev = correr(inyeccion.falla_pos(panel, semana.origen, dias=dias))
            de_tienda = next(m for m in _en_segmento(rev.medidas, "tienda", tienda))
            potencia_falla.append(
                {
                    "origen": str(semana.origen),
                    "tienda": tienda,
                    "dias": dias,
                    "pct_sin_units_sold": _r(de_tienda["valor"]),
                    "p_valor": _sig(de_tienda["p_valor"]),
                    "detectado_por": sorted(
                        {*_detectado_por(rev.alertas, "tienda", tienda)}
                        | {
                            f"tasas_globales:{a['medida']}"
                            for a in rev.alertas
                            if a["chequeo"] == "tasas_globales" and a["medida"] == "pct_sin_units_sold"
                        }
                    ),
                }
            )

    def resumen_salto(cambio: int, dimension: str) -> dict[str, Any]:
        casos = [c for c in potencia_salto if c["cambio_pct"] == cambio and c["segmento"] == dimension]
        detectan = Counter(d for c in casos for d in c["detectado_por"])
        return {
            "segmento": dimension,
            "cambio_pct": cambio,
            "casos": len(casos),
            "nivel_demanda": detectan["nivel_demanda"],
            **{f"sesgo_localizado:{m}": detectan[f"sesgo_localizado:{m}"] for m in modelos},
            "alguno": sum(bool(c["detectado_por"]) for c in casos),
        }

    return {
        "semanas": [{"origen": str(s.origen), "periodo": s.periodo} for s in semanas_val],
        "escenarios": escenarios,
        "todos_detectados": all(e["detectado"] for e in escenarios),
        "potencia": {
            "salto_nivel": potencia_salto,
            "falla_pos": potencia_falla,
            "resumen_salto_nivel": [resumen_salto(round(100 * c), d) for d in POTENCIA for c in MAGNITUDES],
        },
    }


def monitorear(
    panel: pl.DataFrame,
    pronosticos: pl.DataFrame,
    backtest: pl.DataFrame,
    cortes: Sequence[date],
    eventos: Mapping[str, str],
    seleccion: Mapping[str, Any],
    u: Umbrales,
    semanas_validacion: Sequence[date] = SEMANAS_VALIDACION,
) -> dict[str, Any]:
    """El contenido de `monitor.json`, sin la fuente.

    `pronosticos` son los del campeón y el retador en la calibración y en el backtest (solo se usa `yhat`);
    `backtest` son los del backtest con su real, para la vigilancia del retador.
    """
    campeon, retador = campeon_y_retador(seleccion)
    modelos = [campeon, retador]
    todas = semanas(panel, cortes, u)
    con_pronostico = set(pronosticos["origen"].unique().to_list())

    revisiones = {
        s.origen: revisar(panel, pronosticos, s.origen, u) for s in todas if s.periodo == "calibracion"
    }
    calibradas = [r for o, r in revisiones.items() if o in con_pronostico]
    banda_modelos = bandas(calibradas, u)
    banda = {m: b["banda_pct"] for m, b in banda_modelos.items()}
    for s in todas:
        if s.periodo == "backtest":
            revisiones[s.origen] = revisar(panel, pronosticos, s.origen, u, banda)
            if revisiones[s.origen].evento != eventos.get(str(s.origen)):
                raise ValueError(f"el evento del corte {s.origen} no coincide con el de metricas.json")

    vigilancia = vigilar_retador(backtest, campeon, retador, u.horizonte, u)
    reevaluar = {f["corte"]: f["reevaluar"] for f in vigilancia}
    semanas_json = []
    for s in todas:
        rev = revisiones[s.origen]
        alertas = rev.alertas
        semanas_json.append(
            {
                "origen": str(s.origen),
                "periodo": s.periodo,
                "desde": str(s.origen + timedelta(days=1)),
                "hasta": str(s.origen + timedelta(days=u.horizonte)),
                "evento": rev.evento,
                "dias_especiales": rev.especiales,
                "precedentes": rev.precedentes,
                "chequeos_corridos": sorted({m["chequeo"] for m in rev.medidas}),
                "alertas": [_alerta_json(a) for a in alertas],
                "desempeno": _desempeno_json(rev.desempeno),
                "recomendaciones": recomendar(
                    alertas, rev.sin_precedente, reevaluar.get(str(s.origen), False)
                ),
            }
        )

    con_dm = [f for f in vigilancia if f["dm"] is not None]
    final = con_dm[-1]["dm"] if con_dm and con_dm[-1] is vigilancia[-1] else None
    minima = min(con_dm, key=lambda f: f["dm"]["p_valor"]) if con_dm else None
    coincide = _coincide(final, seleccion, campeon, retador, len(vigilancia))
    if coincide is False:
        raise ValueError("la DM acumulada al último corte no coincide con la de seleccion.json")
    periodo_de = {s.origen: s.periodo for s in todas}
    fuera = [str(o) for o in semanas_validacion if o not in periodo_de]
    if fuera:
        raise ValueError(
            f"las semanas de validación {', '.join(fuera)} no están entre las semanas del monitor"
        )
    return {
        "campeon": campeon,
        "retador": retador,
        "configuracion": {
            "umbrales": u.a_json(),
            "referencia": (
                f"los {u.dias_referencia} días que terminan en T, sin días especiales (Buen Fin, temporada "
                "navideña, Semana Santa y festivos)"
            ),
            "segmentos": "global y cada nivel de formato, región, NSE y categoría",
            "bonferroni": bonferroni(list(revisiones.values()), vigilancia, u),
            "banda_wape": {
                m: {k: _r(v) if k != "semanas" else v for k, v in b.items()} for m, b in banda_modelos.items()
            },
            "chequeos": {
                c: {"capa": capa, "cuando": cuando, "medida": medida}
                for c, (capa, cuando, medida) in CHEQUEOS.items()
            },
            "esperadas_por_calendario": (
                "nivel, ticket y desempeño, si la semana tiene un día especial o evento buen_fin, navidad o "
                "post_buen_fin; calidad y promoción siempre son inesperadas"
            ),
        },
        "semanas": semanas_json,
        "resumen": _resumen(semanas_json),
        "vigilancia_retador": {
            "por_corte": _vigilancia_json(vigilancia),
            "final": None
            if final is None
            else {
                "p_valor": round(final["p_valor"], 6),
                "a_favor_de": final["a_favor_de"],
                "cortes_necesarios": vigilancia[-1]["cortes_necesarios"],
            },
            "minima": None
            if minima is None
            else {"corte": minima["corte"], "p_valor": round(minima["dm"]["p_valor"], 6)},
            "coincide_con_seleccion": coincide,
        },
        "validacion": validar(
            panel,
            pronosticos,
            [Semana(o, periodo_de[o]) for o in semanas_validacion],
            banda,
            modelos,
            u,
        ),
    }


def ejecutar(
    panel_path: Path,
    pronosticos_path: Path,
    metricas_path: Path,
    seleccion_path: Path,
    calibracion_path: Path,
    salida_path: Path,
    umbrales: Umbrales | None = None,
    semanas_validacion: Sequence[date] = SEMANAS_VALIDACION,
) -> dict[str, Any]:
    """Lee el panel, los artefactos de E4 y la calibración y escribe `monitor.json`."""
    u = umbrales or Umbrales()
    seleccion = json.loads(seleccion_path.read_text())
    verificar_fuente(seleccion, metricas_path, pronosticos_path)
    campeon, retador = campeon_y_retador(seleccion)
    protocolo = json.loads(metricas_path.read_text())["protocolo"]
    if protocolo["horizonte"] != u.horizonte:
        raise ValueError("el horizonte del monitor no es el del backtest")
    cortes = [date.fromisoformat(o) for o in protocolo["origenes"]]

    panel = pl.read_parquet(panel_path).sort([*SERIE, "date"])
    verificar_malla(panel)
    backtest = pl.read_parquet(pronosticos_path).filter(pl.col("modelo").is_in([campeon, retador]))
    calibracion_pron = calibracion.leer(
        calibracion_path, panel_path, seleccion_path, [campeon, retador], cortes[0], u.horizonte
    )
    columnas = ["modelo", "origen", *SERIE, "date", "yhat"]
    orden = pl.col("modelo").cast(pl.Enum([campeon, retador]))
    pronosticos = pl.concat([calibracion_pron.select(columnas), backtest.select(columnas)]).sort(
        orden, "origen", *SERIE, "date"
    )

    monitor = {
        "fuente": {
            "panel_sha256": sha256(panel_path),
            "pronosticos_sha256": sha256(pronosticos_path),
            "metricas_sha256": sha256(metricas_path),
            "seleccion_sha256": sha256(seleccion_path),
            "calibracion_sha256": sha256(calibracion_path),
        },
        **monitorear(
            panel,
            pronosticos,
            backtest,
            cortes,
            protocolo["evento_por_corte"],
            seleccion,
            u,
            semanas_validacion,
        ),
    }
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    salida_path.write_text(json.dumps(monitor, indent=2, ensure_ascii=False) + "\n")
    return monitor
