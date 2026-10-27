"""La regla de campeón, fijada en la spec de E4 antes de ver sus resultados, su sensibilidad y la salida.

Un retador reemplaza al incumbente solo si cumple las cuatro condiciones:
1. gana en el global: WAPE semanal global menor que el del incumbente;
2. la diferencia es significativa: Diebold-Mariano global contra el incumbente, con p < α y a su favor;
3. no pierde en las semanas normales: WAPE semanal en `resto` ≤ el del incumbente + el margen;
4. no sesga ningún subgrupo de tiendas: |sesgo semanal| ≤ el tope en el global y en cada nivel de formato,
   región, NSE y categoría. Los eventos quedan fuera: en el Buen Fin todos subpronostican cerca de la mitad.

Si varios pasan, gana el de menor WAPE semanal global; si empatan, el primero del registro. Si ninguno pasa,
el campeón sigue siendo el incumbente. Los umbrales viven en `Regla` y se escriben en la salida: cambiarlos no
exige repetir el backtest, solo `make seleccion`.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from demanda.datos.fuente import sha256
from demanda.seleccion.diebold_mariano import (
    MIN_CORTES,
    PruebaDM,
    diebold_mariano,
    errores_semanales,
    perdida_por_corte,
)

NORMAL = "resto"  # el nivel de evento de las semanas normales
TOLERANCIA_PP = 1e-3  # entre la WAPE de metricas.json (redondeada a 4 decimales) y la recalculada
CRITERIOS = {
    "gana_global": "WAPE semanal global menor que el del incumbente",
    "significativa": "Diebold-Mariano global contra el incumbente con p < α y la diferencia a su favor",
    "no_pierde_en_resto": "WAPE semanal en semanas normales (resto) ≤ el del incumbente + el margen",
    "sin_sesgo": "|sesgo semanal| ≤ el tope en el global y en cada nivel de formato, región, NSE y categoría",
}


@dataclass(frozen=True)
class Regla:
    """Los umbrales de la regla y de su sensibilidad, fijados de antemano."""

    incumbente: str = "sistema_actual"
    baselines: tuple[str, ...] = ("naive_estacional", "ma7", "autoets")
    alfa: float = 0.05
    margen_resto_pp: float = 0.5
    sesgo_max_pct: float = 5.0
    subgrupos: tuple[str, ...] = ("global", "formato", "region", "nse", "categoria")
    margenes_sensibilidad_pp: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    alfas_sensibilidad: tuple[float, ...] = (0.05, 0.10)


class Evidencia(NamedTuple):
    """Lo que la regla mira de cada modelo."""

    wape_global: float
    wape_resto: float
    sesgo_max: float  # el mayor |sesgo semanal| entre los subgrupos
    sesgo_peor: str  # el subgrupo donde ocurre, como `segmento=nivel`
    dm_global: PruebaDM | None  # contra el incumbente; None en el incumbente


class Criterio(NamedTuple):
    valor: float
    umbral: float
    cumple: bool


def _redondeo(x: float) -> float:
    return round(x, 4)


def criterios(
    ev: Evidencia, incumbente: Evidencia, regla: Regla, margen: float, alfa: float
) -> dict[str, Criterio]:
    """Los cuatro criterios de un retador. Se comparan con 4 decimales, los de `metricas.json`."""
    if ev.dm_global is None:
        raise ValueError("un retador necesita su prueba Diebold-Mariano contra el incumbente")
    dm = ev.dm_global
    umbral_resto = incumbente.wape_resto + margen
    return {
        "gana_global": Criterio(
            ev.wape_global,
            incumbente.wape_global,
            _redondeo(ev.wape_global) < _redondeo(incumbente.wape_global),
        ),
        "significativa": Criterio(dm.p_valor, alfa, dm.p_valor < alfa and dm.media < 0),
        "no_pierde_en_resto": Criterio(
            ev.wape_resto, umbral_resto, _redondeo(ev.wape_resto) <= _redondeo(umbral_resto)
        ),
        "sin_sesgo": Criterio(
            ev.sesgo_max, regla.sesgo_max_pct, _redondeo(ev.sesgo_max) <= _redondeo(regla.sesgo_max_pct)
        ),
    }


def elegir(
    evidencias: Mapping[str, Evidencia], regla: Regla, margen: float | None = None, alfa: float | None = None
) -> tuple[str, list[str]]:
    """El campeón y los retadores que pasan, con los umbrales de la regla o los de una sensibilidad.

    `evidencias` va en el orden del registro, que es el del desempate.
    """
    margen = regla.margen_resto_pp if margen is None else margen
    alfa = regla.alfa if alfa is None else alfa
    incumbente = evidencias[regla.incumbente]
    pasan = [
        m
        for m, ev in evidencias.items()
        if m != regla.incumbente
        and all(c.cumple for c in criterios(ev, incumbente, regla, margen, alfa).values())
    ]
    if not pasan:
        return regla.incumbente, []
    orden = list(evidencias)
    return min(pasan, key=lambda m: (_redondeo(evidencias[m].wape_global), orden.index(m))), pasan


def sensibilidad(evidencias: Mapping[str, Evidencia], regla: Regla) -> list[dict[str, Any]]:
    """Qué retadores pasarían, y quién sería campeón, con cada margen en `resto` y cada α."""
    filas = []
    for margen in regla.margenes_sensibilidad_pp:
        for alfa in regla.alfas_sensibilidad:
            campeon, pasan = elegir(evidencias, regla, margen, alfa)
            filas.append({"margen_resto_pp": margen, "alfa": alfa, "pasan": pasan, "campeon": campeon})
    return filas


def _segmentos(metricas: Mapping[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(metricas["segmentos"]).filter(pl.col("granularidad") == "semanal")


def _valor(semanal: pl.DataFrame, modelo: str, segmento: str, nivel: str, metrica: str) -> float:
    fila = semanal.filter(
        (pl.col("modelo") == modelo)
        & (pl.col("segmento") == segmento)
        & (pl.col("nivel") == nivel)
        & (pl.col("metrica") == metrica)
    )
    if fila.height != 1:
        raise ValueError(f"metricas.json no tiene {metrica} semanal de {modelo} en {segmento}={nivel}")
    return float(fila["valor"].item())


def _sesgo_peor(semanal: pl.DataFrame, modelo: str, subgrupos: Sequence[str]) -> tuple[float, str]:
    sesgos = (
        semanal.filter(
            (pl.col("modelo") == modelo)
            & pl.col("segmento").is_in(subgrupos)
            & (pl.col("metrica") == "sesgo_pct")
        )
        .with_columns(pl.col("valor").abs().alias("abs"))
        .sort("abs", "segmento", "nivel", descending=[True, False, False])
    )
    if sesgos.is_empty():
        raise ValueError(f"metricas.json no tiene el sesgo semanal de {modelo}")
    peor = sesgos.row(0, named=True)
    return float(peor["abs"]), f"{peor['segmento']}={peor['nivel']}"


def _wape_recalculada(errores: pl.DataFrame) -> dict[str, float]:
    por_modelo = errores.group_by("modelo").agg(
        (100 * pl.col("error").abs().sum() / pl.col("y").sum()).alias("w")
    )
    return dict(por_modelo.iter_rows())


def _comparar(
    errores: pl.DataFrame, a: str, b: str, normales: Sequence[str]
) -> tuple[PruebaDM, list[dict[str, Any]]]:
    """DM de A contra B en el global y en las semanas normales. Devuelve la global, que es la que usa la
    regla, y las dos para la salida, con la diferencia de WAPE semanal en pp sobre los mismos pares.

    La de las semanas normales es informativa: con menos de `MIN_CORTES` cortes normales queda en nulo.
    """
    perdida = perdida_por_corte(errores, a, b)
    dm_global = diebold_mariano(perdida["d"].to_list())
    filas = []
    for segmento, cortes in (
        ("global", perdida),
        (NORMAL, perdida.filter(pl.col("origen").cast(pl.String).is_in(normales))),
    ):
        dm = dm_global if segmento == "global" else None
        if segmento == NORMAL and cortes.height >= MIN_CORTES:
            dm = diebold_mariano(cortes["d"].to_list())
        diferencia = (
            cortes.select(100 * pl.col("d").sum() / pl.col("y").sum()).item() if cortes.height else None
        )
        filas.append(
            {
                "a": a,
                "b": b,
                "segmento": segmento,
                "n_cortes": cortes.height,
                "diferencia_wape_pp": None if diferencia is None else _redondeo(diferencia),
                "media_d": None if dm is None else _redondeo(dm.media),
                "estadistico": None if dm is None else _redondeo(dm.estadistico),
                "p_valor": None if dm is None else round(dm.p_valor, 6),
                "a_favor_de": None if dm is None else (a if dm.media < 0 else b),
            }
        )
    return dm_global, filas


def _verificar_corrida(metricas: Mapping[str, Any], pronosticos: pl.DataFrame, errores: pl.DataFrame) -> None:
    """`metricas.json` y `pronosticos.parquet` deben salir de la misma corrida: los mismos modelos, los mismos
    cortes (también en `evento_por_corte`) y la misma WAPE semanal global."""
    protocolo = metricas["protocolo"]
    origenes = set(protocolo["origenes"])
    problemas = []
    if set(pronosticos["modelo"].unique().to_list()) != set(protocolo["modelos"]):
        problemas.append("los modelos no coinciden")
    if {str(o) for o in pronosticos["origen"].unique().to_list()} != origenes:
        problemas.append("los cortes no coinciden")
    if set(protocolo["evento_por_corte"]) != origenes:
        problemas.append("evento_por_corte no tiene los mismos cortes")
    semanal, recalculada = _segmentos(metricas), _wape_recalculada(errores)
    for m in protocolo["modelos"]:
        diferencia = abs(
            recalculada.get(m, float("nan")) - _valor(semanal, m, "global", "global", "wape_pct")
        )
        if not diferencia <= TOLERANCIA_PP:  # un NaN también es inconsistencia
            problemas.append(f"la WAPE semanal global de {m} no coincide")
    if problemas:
        raise ValueError(
            "metricas.json y pronosticos.parquet no son de la misma corrida: " + "; ".join(problemas)
        )


def _criterios_json(evaluados: Mapping[str, Criterio]) -> dict[str, dict[str, Any]]:
    return {
        n: {"valor": _redondeo(c.valor), "umbral": _redondeo(c.umbral), "cumple": c.cumple}
        for n, c in evaluados.items()
    }


def _justificacion(
    campeon: str,
    evidencias: Mapping[str, Evidencia],
    evaluados: Mapping[str, dict[str, Criterio]],
    regla: Regla,
) -> str:
    """La frase que justifica el resultado, generada con plantilla."""
    inc = evidencias[regla.incumbente]
    if campeon != regla.incumbente:
        ev, dm = evidencias[campeon], evidencias[campeon].dm_global
        assert dm is not None
        return (
            f"{campeon} reemplaza a {regla.incumbente}: WAPE semanal global de {ev.wape_global:.2f} % contra "
            f"{inc.wape_global:.2f} % (Diebold-Mariano p = {dm.p_valor:.3f}), {ev.wape_resto:.2f} % contra "
            f"{inc.wape_resto:.2f} % en semanas normales y un |sesgo| de a lo más {ev.sesgo_max:.2f} % por "
            "subgrupo."
        )
    if not evaluados:
        return f"No hay retadores: {regla.incumbente} sigue como campeón."
    mejor = min(evaluados, key=lambda m: (_redondeo(evidencias[m].wape_global), list(evidencias).index(m)))
    fallas = []
    for nombre, c in evaluados[mejor].items():
        if c.cumple:
            continue
        if nombre == "significativa":
            fallas.append(f"la diferencia no es significativa (p = {c.valor:.3f}, α = {c.umbral:.2f})")
        elif nombre == "gana_global":
            fallas.append(f"no gana en el global ({c.valor:.2f} % contra {c.umbral:.2f} %)")
        elif nombre == "no_pierde_en_resto":
            fallas.append(f"pierde en semanas normales ({c.valor:.2f} % contra un tope de {c.umbral:.2f} %)")
        else:
            fallas.append(f"sesga un subgrupo ({evidencias[mejor].sesgo_peor}: |sesgo| de {c.valor:.2f} %)")
    return (
        f"Ningún retador cumple las cuatro condiciones, así que {regla.incumbente} sigue como campeón. "
        f"El de menor WAPE semanal global, {mejor} ({evidencias[mejor].wape_global:.2f} % contra "
        f"{inc.wape_global:.2f} %), falla porque {'; '.join(fallas)}."
    )


def seleccionar(metricas: Mapping[str, Any], pronosticos: pl.DataFrame, regla: Regla) -> dict[str, Any]:
    """Aplica la regla a las salidas del backtest: el contenido de `seleccion.json`, sin la fuente."""
    protocolo = metricas["protocolo"]
    modelos: list[str] = protocolo["modelos"]
    if regla.incumbente not in modelos:
        raise ValueError(f"el backtest no incluye al incumbente {regla.incumbente}")
    eventos: dict[str, str] = protocolo.get("evento_por_corte") or {}
    if not eventos:
        raise ValueError("metricas.json no tiene `evento_por_corte`; corre de nuevo `demanda backtest`")
    normales = [c for c, e in eventos.items() if e == NORMAL]

    semanal = _segmentos(metricas)
    errores = errores_semanales(pronosticos, protocolo["horizonte"])
    _verificar_corrida(metricas, pronosticos, errores)

    # Cada retador contra el incumbente, y cada modelo de E4 contra el mejor baseline
    pruebas: list[dict[str, Any]] = []
    contra_incumbente: dict[str, PruebaDM] = {}
    for m in modelos:
        if m != regla.incumbente:
            contra_incumbente[m], filas = _comparar(errores, m, regla.incumbente, normales)
            pruebas += filas
    evidencias = {
        m: Evidencia(
            _valor(semanal, m, "global", "global", "wape_pct"),
            _valor(semanal, m, "evento", NORMAL, "wape_pct"),
            *_sesgo_peor(semanal, m, regla.subgrupos),
            contra_incumbente.get(m),
        )
        for m in modelos
    }
    con_baseline = [b for b in regla.baselines if b in evidencias]
    mejor_baseline = min(con_baseline, key=lambda b: evidencias[b].wape_global) if con_baseline else None
    if mejor_baseline is not None:
        for m in modelos:
            if m not in (regla.incumbente, *regla.baselines):
                pruebas += _comparar(errores, m, mejor_baseline, normales)[1]

    incumbente = evidencias[regla.incumbente]
    evaluados = {
        m: criterios(ev, incumbente, regla, regla.margen_resto_pp, regla.alfa)
        for m, ev in evidencias.items()
        if m != regla.incumbente
    }
    campeon, _ = elegir(evidencias, regla)
    return {
        "regla": {**asdict(regla), "criterios": CRITERIOS},
        "incumbente": regla.incumbente,
        "mejor_baseline": None
        if mejor_baseline is None
        else {"modelo": mejor_baseline, "wape_semanal_global_pct": evidencias[mejor_baseline].wape_global},
        "cortes": {"total": len(eventos), NORMAL: len(normales)},
        "pruebas_dm": pruebas,
        "modelos": {
            m: {
                "wape_semanal_global_pct": ev.wape_global,
                "wape_semanal_resto_pct": ev.wape_resto,
                "sesgo_semanal_max_abs_pct": ev.sesgo_max,
                "sesgo_semanal_peor_subgrupo": ev.sesgo_peor,
                "criterios": _criterios_json(evaluados[m]) if m in evaluados else None,
                "pasa": None if m not in evaluados else all(c.cumple for c in evaluados[m].values()),
            }
            for m, ev in evidencias.items()
        },
        "campeon": campeon,
        "justificacion": _justificacion(campeon, evidencias, evaluados, regla),
        "sensibilidad": sensibilidad(evidencias, regla),
    }


def ejecutar(
    metricas_path: Path, pronosticos_path: Path, salida_path: Path, regla: Regla | None = None
) -> dict[str, Any]:
    """Lee las salidas del backtest y escribe `seleccion.json`. Dos corridas sobre los mismos archivos lo
    escriben igual: no lleva fechas de corrida, filas ni series."""
    regla = regla or Regla()
    metricas = json.loads(metricas_path.read_text())
    seleccion = {
        "fuente": {
            "metricas_sha256": sha256(metricas_path),
            "pronosticos_sha256": sha256(pronosticos_path),
        },
        **seleccionar(metricas, pl.read_parquet(pronosticos_path), regla),
    }
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    salida_path.write_text(json.dumps(seleccion, indent=2, ensure_ascii=False) + "\n")
    return seleccion
