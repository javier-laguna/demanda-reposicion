"""La etapa de decisión: de los pronósticos del backtest y de la calibración a `impacto.json`.

- Tres políticas: el pronóstico puntual del campeón y el newsvendor sobre el campeón y sobre el retador. De
  la primera a la segunda va el valor de la regla de decisión; de la segunda a la tercera, el del mejor
  pronóstico.
- El impacto en los pares completos de los cortes del backtest: en el global, por evento, por categoría y por
  formato.
- La Diebold-Mariano de E4 sobre el costo por corte, en el global y en `resto`, y la sensibilidad a la
  fracción crítica.

No lleva filas ni series, y dos corridas sobre los mismos archivos lo escriben igual.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from demanda.backtest import calibracion
from demanda.datos.fuente import sha256
from demanda.decision import newsvendor
from demanda.decision.newsvendor import NORMAL
from demanda.decision.supuestos import Supuestos
from demanda.seleccion.campeon import campeon_y_retador, verificar_fuente
from demanda.seleccion.diebold_mariano import MIN_CORTES, diebold_mariano

SEGMENTOS = {"global": None, "evento": "evento", "categoria": "category", "formato": "store_format"}
SENSIBLES = ("costo", "fill_rate_pct", "nivel_servicio_pct", "sobreinventario_pct", "quiebre_pct")
DEFINICIONES = {
    "r": "Σy / Σŷ por serie y corte, en los pares completos",
    "q": (
        "cuantil ponderado de r en la fracción crítica, con peso Σŷ y sin interpolar; ventana: las "
        "semanas de calibración y los cortes anteriores del backtest con evento resto"
    ),
    "sobreinventario": "Σ max(S − y, 0), en unidades; _pct es sobre la demanda Σy",
    "quiebre": "Σ max(y − S, 0), en unidades; _pct es sobre la demanda Σy",
    "nivel_servicio_pct": "% de pares serie × corte sin quiebre",
    "fc_objetivo_pct": "la fracción crítica media de los pares: el nivel de servicio objetivo",
    "fill_rate_pct": "100 · (1 − quiebre / demanda)",
    "costo": "Σ Cu · quiebre + Co · sobreinventario, en pesos",
    "stock_sobre_demanda": "Σ S / Σ y",
    "prueba_costo": "Diebold-Mariano con d_t = costo_a,t − costo_b,t por corte; negativa favorece a a",
}


def _r(x: float | None, digitos: int = 4) -> float | None:
    return None if x is None else round(float(x), digitos)


def evaluar(
    pares_calibracion: pl.DataFrame,
    pares_backtest: pl.DataFrame,
    eventos: Mapping[date, str],
    campeon: str,
    retador: str,
    costos: pl.DataFrame,
    horizonte: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Los multiplicadores por corte y los resultados por par de las tres políticas, con unos costos dados."""
    pares = pl.concat(
        [
            pares_calibracion.with_columns(pl.lit(True).alias("normal")),
            pares_backtest.with_columns(
                pl.col("origen").replace_strict(eventos, return_dtype=pl.String).eq(NORMAL).alias("normal")
            ),
        ]
    )
    fracciones = dict(costos.select("category", "fc").iter_rows())
    mult = newsvendor.multiplicadores(pares, sorted(eventos), fracciones, horizonte)
    niveles = newsvendor.niveles_objetivo(pares_backtest, mult, campeon, retador)
    return mult, newsvendor.resultados(niveles, costos)


def _tabla(res: pl.DataFrame, politicas: Sequence[str]) -> list[dict[str, Any]]:
    partes = []
    for segmento, columna in SEGMENTOS.items():
        nivel = pl.lit("global") if columna is None else pl.col(columna)
        d = res.with_columns(pl.lit(segmento).alias("segmento"), nivel.alias("nivel"))
        partes.append(newsvendor.metricas(d, ["segmento", "nivel", "politica"]))
    tabla = (
        pl.concat(partes)
        .with_columns(
            pl.col("segmento").cast(pl.Enum(list(SEGMENTOS))),
            pl.col("politica").cast(pl.Enum(list(politicas))),
        )
        .sort("segmento", "nivel", "politica")
        .with_columns(pl.col("segmento", "politica").cast(pl.String))
    )
    filas = []
    for f in tabla.iter_rows(named=True):
        filas.append(
            {
                "segmento": f["segmento"],
                "nivel": f["nivel"],
                "politica": f["politica"],
                "pares": f["pares"],
                "demanda": _r(f["demanda"], 2),
                "stock": _r(f["stock"], 2),
                "sobreinventario": _r(f["sobreinventario"], 2),
                "quiebre": _r(f["quiebre"], 2),
                **{c: _r(f[c]) for c in ("sobreinventario_pct", "quiebre_pct", "fill_rate_pct")},
                "nivel_servicio_pct": _r(f["nivel_servicio_pct"]),
                "fc_objetivo_pct": _r(f["fc_objetivo_pct"]),
                "costo": _r(f["costo"], 2),
                "stock_sobre_demanda": _r(f["stock_sobre_demanda"]),
            }
        )
    return filas


def _pruebas(
    res: pl.DataFrame, comparaciones: Sequence[tuple[str, str]], normales: Sequence[date]
) -> list[dict[str, Any]]:
    """La DM sobre el costo por corte de cada comparación, en el global y en `resto`. Con menos de
    `MIN_CORTES` cortes, o si d_t no varía, la prueba queda en nulo; sin costo en b, también la diferencia
    relativa."""
    por_corte = (
        newsvendor.costo_por_corte(res).pivot("politica", index="origen", values="costo").sort("origen")
    )
    filas = []
    for a, b in comparaciones:
        for segmento, cortes in (
            ("global", por_corte),
            (NORMAL, por_corte.filter(pl.col("origen").is_in(normales))),
        ):
            d = (cortes[a] - cortes[b]).to_list()
            dm = diebold_mariano(d) if len(d) >= MIN_CORTES and np.std(d) > 0 else None
            costo_a, costo_b = float(cortes[a].sum()), float(cortes[b].sum())
            filas.append(
                {
                    "a": a,
                    "b": b,
                    "segmento": segmento,
                    "n_cortes": len(d),
                    "costo_a": _r(costo_a, 2),
                    "costo_b": _r(costo_b, 2),
                    "diferencia": _r(costo_a - costo_b, 2),
                    "diferencia_pct": _r(100 * (costo_a - costo_b) / costo_b) if costo_b else None,
                    "media_d": None if dm is None else _r(dm.media, 2),
                    "estadistico": None if dm is None else _r(dm.estadistico),
                    "p_valor": None if dm is None else round(dm.p_valor, 6),
                    "a_favor_de": None if dm is None else (a if dm.media < 0 else b),
                }
            )
    return filas


def _resumen(res: pl.DataFrame, politicas: Sequence[str]) -> dict[str, dict[str, float | None]]:
    """Por política, en el orden de `politicas`, las métricas de la sensibilidad."""
    orden = pl.col("politica").cast(pl.Enum(list(politicas)))
    tabla = newsvendor.metricas(res, ["politica"]).sort(orden)
    return {
        f["politica"]: {c: _r(f[c], 2 if c == "costo" else 4) for c in SENSIBLES}
        for f in tabla.iter_rows(named=True)
    }


def _orden(resumen: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Las políticas de menor a mayor costo."""
    return sorted(resumen, key=lambda p: (resumen[p]["costo"], p))


def _conclusion(pruebas: Sequence[Mapping[str, Any]], se_mantiene: bool, n: int) -> str:
    """La frase de conclusión, hecha con plantilla, con las dos comparaciones globales."""
    globales = [p for p in pruebas if p["segmento"] == "global"]
    regla, pronostico = globales[1], globales[0]

    def comparar(p: Mapping[str, Any]) -> str:
        valor_p = "—" if p["p_valor"] is None else f"{p['p_valor']:.3f}"
        if p["diferencia"] == 0:
            return f"lo mismo que {p['b']} (Diebold-Mariano p = {valor_p})"
        direccion = "menos" if p["diferencia"] < 0 else "más"
        cuanto = (
            f"{abs(p['diferencia']):,.2f} pesos"
            if p["diferencia_pct"] is None
            else f"{abs(p['diferencia_pct']):.1f} %"
        )
        return f"{cuanto} {direccion} que {p['b']} (Diebold-Mariano p = {valor_p})"

    return (
        f"Con los costos supuestos, {regla['a']} cuesta {comparar(regla)}, y {pronostico['a']} cuesta "
        f"{comparar(pronostico)}. El orden de las tres políticas por costo "
        f"{'se mantiene' if se_mantiene else 'cambia'} en las {n} fracciones críticas de la sensibilidad."
    )


def decidir(
    backtest: pl.DataFrame,
    calibracion_pron: pl.DataFrame,
    panel: pl.DataFrame,
    eventos: Mapping[date, str],
    campeon: str,
    retador: str,
    horizonte: int,
    supuestos: Supuestos,
) -> dict[str, Any]:
    """El contenido de `impacto.json`, sin la fuente."""
    cortes = sorted(eventos)
    normales = [c for c in cortes if eventos[c] == NORMAL]
    politicas = list(newsvendor.nombres(campeon, retador))
    pares_bt = newsvendor.pares_semanales(backtest, horizonte)
    pares_cal = newsvendor.pares_semanales(calibracion_pron, horizonte)
    precios = newsvendor.precio_medio(panel, cortes[0])
    costos = newsvendor.costos_unitarios(precios, supuestos)

    mult, res = evaluar(pares_cal, pares_bt, eventos, campeon, retador, costos, horizonte)
    formatos = panel.select("store_id", "store_format").unique()
    res = res.join(formatos, on="store_id", how="left", maintain_order="left").with_columns(
        pl.col("origen").replace_strict(eventos, return_dtype=pl.String).alias("evento")
    )
    comparaciones = [(politicas[2], politicas[1]), (politicas[1], politicas[0])]
    pruebas = _pruebas(res, comparaciones, normales)

    base = _resumen(res, politicas)
    sensibilidad = []
    for fc in supuestos.fracciones_sensibilidad:
        costos_fc = newsvendor.costos_unitarios(precios, supuestos, fc)
        mult_fc, res_fc = evaluar(pares_cal, pares_bt, eventos, campeon, retador, costos_fc, horizonte)
        resumen = _resumen(res_fc, politicas)
        sensibilidad.append(
            {
                "fraccion_critica": fc,
                "global": resumen,
                NORMAL: _resumen(res_fc.filter(pl.col("origen").is_in(normales)), politicas),
                "orden": _orden(resumen),
                "multiplicador_medio": {
                    m: _r(q, 4)
                    for m, q in mult_fc.group_by("modelo").agg(pl.col("q").mean()).sort("modelo").iter_rows()
                },
            }
        )
    se_mantiene = all(s["orden"] == _orden(base) for s in sensibilidad)

    return {
        "campeon": campeon,
        "retador": retador,
        "politicas": newsvendor.nombres(campeon, retador),
        "supuestos": supuestos.a_json(),
        "precios": {
            "hasta": str(cortes[0]),
            "definicion": "Σ amount_total / Σ units_sold por categoría en la historia del primer corte",
            "por_categoria": {
                f["category"]: {
                    "precio_medio": _r(precios[f["category"]], 4),
                    "cu": _r(f["cu"], 4),
                    "co": _r(f["co"], 4),
                    "fraccion_critica": _r(f["fc"], 4),
                }
                for f in costos.iter_rows(named=True)
            },
        },
        "calibracion": {
            "origenes": [str(o) for o in pares_cal["origen"].unique().sort()],
            "cortes": [str(c) for c in cortes],
            "cortes_normales": [str(c) for c in normales],
        },
        "multiplicadores": [
            {
                "corte": str(f["origen"]),
                "modelo": f["modelo"],
                "categoria": f["category"],
                "q": _r(f["q"], 6),
                "pares": f["pares"],
                "semanas": f["semanas"],
            }
            for f in mult.sort("origen", "modelo", "category").iter_rows(named=True)
        ],
        "impacto": _tabla(res, politicas),
        "pruebas_costo": pruebas,
        "sensibilidad": {
            "co": "Cu · (1 − FC) / FC, con Cu fijo y la misma FC en todas las categorías",
            "orden_base": _orden(base),
            "fracciones": sensibilidad,
            "orden_se_mantiene": se_mantiene,
        },
        "conclusion": _conclusion(pruebas, se_mantiene, len(sensibilidad)),
        "definiciones": DEFINICIONES,
    }


def ejecutar(
    pronosticos_path: Path,
    metricas_path: Path,
    seleccion_path: Path,
    calibracion_path: Path,
    panel_path: Path,
    salida_path: Path,
    supuestos: Supuestos | None = None,
) -> dict[str, Any]:
    """Lee los artefactos de E4 y la calibración y escribe `impacto.json`."""
    supuestos = supuestos or Supuestos()
    seleccion = json.loads(seleccion_path.read_text())
    verificar_fuente(seleccion, metricas_path, pronosticos_path)
    campeon, retador = campeon_y_retador(seleccion)
    protocolo = json.loads(metricas_path.read_text())["protocolo"]
    horizonte = protocolo["horizonte"]
    eventos = {date.fromisoformat(c): e for c, e in protocolo["evento_por_corte"].items()}

    backtest = pl.read_parquet(pronosticos_path).filter(pl.col("modelo").is_in([campeon, retador]))
    calibracion_pron = calibracion.leer(
        calibracion_path, panel_path, seleccion_path, [campeon, retador], min(eventos), horizonte
    )
    panel = pl.read_parquet(
        panel_path, columns=["date", "store_id", "category", "amount_total", "units_sold", "store_format"]
    )
    impacto = {
        "fuente": {
            "pronosticos_sha256": sha256(pronosticos_path),
            "metricas_sha256": sha256(metricas_path),
            "seleccion_sha256": sha256(seleccion_path),
            "calibracion_sha256": sha256(calibracion_path),
            "panel_sha256": sha256(panel_path),
        },
        **decidir(backtest, calibracion_pron, panel, eventos, campeon, retador, horizonte, supuestos),
    }
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    salida_path.write_text(json.dumps(impacto, indent=2, ensure_ascii=False) + "\n")
    return impacto
