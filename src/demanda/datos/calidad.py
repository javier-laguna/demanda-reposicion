"""Reporte de calidad: mide cada problema conocido de los datos y deja escrita la decisión que se tomó.

`ejecutar` es la etapa completa: lee los CSV, valida los contratos, construye el panel y escribe el reporte en
JSON. El reporte no lleva filas ni series, solo conteos y estadísticos agregados.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from demanda.datos import contratos, fuente, panel
from demanda.datos.contratos import SERIE, Tablas

SEGMENTOS = {
    "tienda": pl.col("store_id"),
    "categoria": pl.col("category"),
    "mes": pl.col("date").dt.strftime("%Y-%m"),
}
UMBRAL_BLOQUE = 0.95  # una fecha "está en bloque" si al menos este porcentaje de sus filas es nulo
TOLERANCIA_TICKET = 0.01  # avg_ticket viene redondeado a centavos
VENTANAS = range(1, 15)  # días de la media móvil
DESFASES = range(-1, 4)  # día en que termina la ventana, relativo a la fecha de la señal
PESOS = range(-7, 8)
SENAL = "replenishment_signal"


def _pct(parte: float, total: float) -> float:
    return round(100 * parte / total, 2) if total else 0.0


def _uniformidad(df: pl.DataFrame, indicador: pl.Expr) -> dict[str, dict[str, float]]:
    """Mínimo y máximo, por segmento, del porcentaje de filas donde `indicador` es verdadero.

    Un rango estrecho alrededor del promedio global indica que el problema no se concentra en ningún segmento.
    """
    rangos = {}
    for nombre, segmento in SEGMENTOS.items():
        rangos[nombre] = (
            df.group_by(segmento.alias("segmento"))
            .agg((indicador.mean() * 100).alias("pct"))
            .select(pl.col("pct").min().round(2).alias("min"), pl.col("pct").max().round(2).alias("max"))
            .row(0, named=True)
        )
    return rangos


def reporte_malla(p: pl.DataFrame) -> dict[str, Any]:
    faltantes = (
        p.filter("flag_fila_faltante")
        .group_by("store_id", "date")
        .len("filas")
        .sort("store_id", "date")
        .with_columns(
            (pl.col("date").diff().over("store_id") != pl.duration(days=1))
            .fill_null(True)
            .cum_sum()
            .alias("bloque")
        )
    )
    bloques = (
        faltantes.group_by("store_id", "bloque", maintain_order=True)
        .agg(
            pl.col("date").min().cast(pl.String).alias("desde"),
            pl.col("date").max().cast(pl.String).alias("hasta"),
            pl.len().alias("dias"),
            pl.col("filas").sum(),
        )
        .drop("bloque")
    )
    return {
        "esperadas": p.height,
        "observadas": p.height - int(p["flag_fila_faltante"].sum()),
        "faltantes": int(p["flag_fila_faltante"].sum()),
        "bloques": bloques.to_dicts(),
    }


def reporte_conteos(t: pl.DataFrame) -> dict[str, Any]:
    evaluables = t.filter(pl.col("cash_transactions").is_not_null())
    inconsistente = pl.col("cash_transactions") + pl.col("card_transactions") < pl.col("total_transactions")
    n_inconsistentes = evaluables.filter(inconsistente).height

    def ticket_igual_a(denominador: pl.Expr) -> pl.Expr:
        return (
            (pl.col("avg_ticket") - pl.col("amount_total") / denominador).abs() <= TOLERANCIA_TICKET
        ).mean()

    con_ticket = evaluables.filter(pl.col("avg_ticket").is_not_null())
    ticket = con_ticket.select(
        (ticket_igual_a(pl.col("total_transactions")) * 100).round(2).alias("pct_igual_a_monto_entre_total"),
        (ticket_igual_a(pl.col("cash_transactions") + pl.col("card_transactions")) * 100)
        .round(2)
        .alias("pct_igual_a_monto_entre_cash_mas_card"),
    ).row(0, named=True)
    sobreestimacion = con_ticket.filter(inconsistente).select(
        (
            (pl.col("avg_ticket") / (pl.col("amount_total") / pl.col("total_transactions")) - 1).mean() * 100
        ).round(2)
    )
    desviacion_montos = t.select(
        (pl.col("amount_cash") + pl.col("amount_card") - pl.col("amount_total")).abs().max()
    ).item()
    return {
        "regla": "cash_transactions + card_transactions == total_transactions",
        "filas_evaluables": evaluables.height,
        "inconsistentes": n_inconsistentes,
        "pct_inconsistentes": _pct(n_inconsistentes, evaluables.height),
        "uniformidad_pct": _uniformidad(evaluables, inconsistente),
        "avg_ticket": {
            "filas_evaluables": con_ticket.height,
            **ticket,
            "sobreestimacion_media_pct_en_inconsistentes": sobreestimacion.item(),
        },
        "montos_desviacion_max": desviacion_montos,
    }


def reporte_nulos(t: pl.DataFrame) -> dict[str, Any]:
    columnas = [c for c in t.columns if t[c].null_count()]
    por_fecha = t.group_by("date").agg(pl.col(c).is_null().mean() for c in columnas)
    reporte = {}
    for c in columnas:
        es_nulo = pl.col(c).is_null()
        nulos = t[c].null_count()
        fechas_en_bloque = por_fecha.filter(pl.col(c) >= UMBRAL_BLOQUE)["date"]
        en_bloque = t.filter(es_nulo & pl.col("date").is_in(fechas_en_bloque.implode())).height
        entrada: dict[str, Any] = {
            "nulos": nulos,
            "pct": _pct(nulos, t.height),
            "patron": "bloque" if en_bloque >= UMBRAL_BLOQUE * nulos else "disperso",
            "fechas_con_nulos": t.filter(es_nulo)["date"].n_unique(),
        }
        if entrada["patron"] == "bloque":
            entrada["fechas"] = fechas_en_bloque.sort().cast(pl.String).to_list()
        else:
            entrada["uniformidad_pct"] = _uniformidad(t, es_nulo)
        # Porcentaje de los nulos de `c` en los que otra columna también es nula
        entrada["tambien_nulo_pct"] = (
            t.filter(es_nulo).select((pl.col(o).is_null().mean() * 100).round(2) for o in columnas if o != c)
        ).row(0, named=True)
        reporte[c] = entrada
    return reporte


def _centrar(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    """Resta la media de cada serie: así la correlación mide la dinámica y no las diferencias de nivel."""
    return df.drop_nulls(columnas).with_columns(pl.col(c) - pl.col(c).mean().over(SERIE) for c in columnas)


def _correlaciones(df: pl.DataFrame, x: str, y: str) -> tuple[float, float]:
    """(intraserie, agregada)."""
    sub = df.drop_nulls([x, y])
    agregada = sub.select(pl.corr(x, y)).item()
    intraserie = _centrar(sub, [x, y]).select(pl.corr(x, y)).item()
    return round(intraserie, 4), round(agregada, 4)


class Ventana(NamedTuple):
    """Media móvil de `dias` días que va de `desde` a `hasta`, en días relativos a la fecha de la señal."""

    dias: int
    desde: int
    hasta: int
    corr: float
    corr_agregada: float


def reporte_fuga(p: pl.DataFrame) -> dict[str, Any]:
    """¿Qué días de `units_sold` contiene `replenishment_signal`?

    Compara la señal contra medias móviles de 1 a 14 días que terminan entre el día anterior y tres días
    después de la fecha de la señal, y estima por mínimos cuadrados el peso de cada día cercano a esa fecha.
    Si la mejor ventana termina después de la fecha, la señal usa ventas que todavía no ocurrían.
    """
    df = p.select(*SERIE, "date", "units_sold", SENAL).sort([*SERIE, "date"])
    unidades = pl.col("units_sold").cast(pl.Float64)

    ventanas = []
    for dias in VENTANAS:
        for termina in DESFASES:
            media = unidades.rolling_mean(dias, min_samples=1).shift(-termina).over(SERIE).alias("media")
            intraserie, agregada = _correlaciones(df.with_columns(media), SENAL, "media")
            ventanas.append(Ventana(dias, termina - dias + 1, termina, intraserie, agregada))
    ventanas.sort(key=lambda v: v.corr, reverse=True)
    mejor = ventanas[0]

    media_mejor = unidades.rolling_mean(mejor.dias, min_samples=1).shift(-mejor.hasta).over(SERIE).alias("m")
    factor = (
        df.with_columns(media_mejor)
        .drop_nulls([SENAL, "m"])
        .select((pl.col(SENAL) * pl.col("m")).sum() / (pl.col("m") ** 2).sum())
        .item()
    )

    desfases = {f"{j:+d}": unidades.shift(-j).over(SERIE).alias(f"{j:+d}") for j in PESOS}
    matriz = _centrar(df.with_columns(desfases.values()), [SENAL, *desfases])
    pesos, *_ = np.linalg.lstsq(
        matriz.select(list(desfases)).to_numpy(), matriz[SENAL].to_numpy(), rcond=None
    )

    mismo_dia = _correlaciones(df, "units_sold", SENAL)[0]
    de_ayer = _correlaciones(
        df.with_columns(pl.col(SENAL).shift(1).over(SERIE).alias("ayer")), "units_sold", "ayer"
    )[0]
    ma7_hoy = next(v for v in ventanas if v.dias == 7 and v.hasta == 0)
    return {
        "criterio": "correlación intraserie entre la señal y la media móvil de units_sold",
        "mejor_ventana": mejor._asdict(),
        "factor": round(factor, 4),
        "ma7_hasta_el_dia": ma7_hoy._asdict(),
        "ventanas_top": [v._asdict() for v in ventanas[:5]],
        "pesos_por_desfase": {j: round(float(w), 4) for j, w in zip(desfases, pesos, strict=True)},
        "corr_con_units_sold": {"senal_del_mismo_dia": mismo_dia, "senal_del_dia_anterior": de_ayer},
        "incluye_el_dia": mejor.desde <= 0 <= mejor.hasta,
        "usa_informacion_futura": mejor.hasta > 0,
        # Con origen T se conocen las ventas hasta T: un valor fechado en d usa ventas hasta d + hasta
        "rezago_minimo_desde_el_origen": max(mejor.hasta, 0),
    }


def reporte_dimensiones(tablas: Tablas, p: pl.DataFrame) -> dict[str, Any]:
    t, tiendas, _ = tablas
    primer_anio = p.select(pl.col("date").min().dt.year()).item()
    return {
        "filas": t.height,
        "tiendas": t["store_id"].n_unique(),
        "tiendas_en_catalogo": tiendas.height,
        "tiendas_abiertas_antes_del_periodo": tiendas.filter(pl.col("opening_year") < primer_anio).height,
        "categorias": t["category"].n_unique(),
        "dias": p["date"].n_unique(),
        "desde": str(p["date"].min()),
        "hasta": str(p["date"].max()),
        "series": p.select(SERIE).n_unique(),
    }


def construir_reporte(tablas: Tablas, p: pl.DataFrame, hashes: dict[str, str]) -> dict[str, Any]:
    fuga = reporte_fuga(p)
    rezago = fuga["rezago_minimo_desde_el_origen"]
    return {
        "fuente": hashes,
        "contratos": {**dict.fromkeys(contratos.ESQUEMAS, "ok"), "uniones": "ok"},
        "dimensiones": reporte_dimensiones(tablas, p),
        "malla": reporte_malla(p),
        "conteos": reporte_conteos(tablas.transacciones),
        "nulos": reporte_nulos(tablas.transacciones),
        "fuga_replenishment_signal": fuga,
        "decisiones": [
            "Objetivo: units_sold. Si falta, la fila se excluye de entrenamiento y evaluación; no se imputa.",
            "Las filas faltantes de la malla no se imputan: quedan en el panel con flag_fila_faltante.",
            "Los conteos inconsistentes no se corrigen: se marcan con flag_conteo_inconsistente.",
            f"replenishment_signal: con origen T solo se usan valores fechados en T - {rezago} o antes.",
        ],
    }


def ejecutar(raw_dir: Path, panel_path: Path, reporte_path: Path) -> dict[str, Any]:
    """Valida los CSV de `raw_dir`, escribe el panel y el reporte, y devuelve el reporte."""
    tablas = contratos.validar(fuente.leer(raw_dir))
    p = panel.construir(tablas)
    hashes = {archivo: fuente.sha256(raw_dir / archivo) for archivo in fuente.ARCHIVOS}
    reporte = construir_reporte(tablas, p, hashes)

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    p.write_parquet(panel_path)
    reporte_path.parent.mkdir(parents=True, exist_ok=True)
    reporte_path.write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n")
    return reporte
