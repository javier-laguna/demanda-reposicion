"""La etapa completa: lee el panel y el reporte de calidad, corre el backtest y escribe sus salidas.

- `pronosticos.parquet`: modelo, origen, serie, fecha, h, real y pronósticos;
- `metricas.json`: fuente, protocolo, cobertura, conciliación con E2, tablas por segmento y definiciones. No
  lleva filas ni series, y dos corridas sobre los mismos datos lo escriben igual;
- un run de MLflow por modelo.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from demanda.backtest import bucle, conciliacion, segmentos, seguimiento
from demanda.backtest.protocolo import Protocolo, ultima_fecha
from demanda.datos.contratos import LLAVE, SERIE
from demanda.datos.fuente import sha256
from demanda.modelos import Contexto, Modelo, crear

DEFINICIONES = {
    "metricas": {
        "wape_pct": "100 · Σ|y − ŷ| / Σy",
        "sesgo_pct": "100 · Σ(ŷ − y) / Σy; positivo es sobrepronóstico",
        "mase": (
            "por serie y corte, el MAE del horizonte entre el MAE del naive estacional 7 en la historia del "
            "corte; se promedia sobre series × cortes"
        ),
        "pinball_<τ>_pct": (
            "100 · Σ ρ_τ(y − q_τ) / Σy, con ρ_τ(u) = max(τu, (τ − 1)u) y q_0.5 = ŷ; en un modelo puntual "
            "solo existe pinball_50 = ½ WAPE"
        ),
        "pinball_medio_pct": "promedio de pinball sobre los cuantiles; solo en modelos con intervalo",
        "cobertura_80_pct": "100 · proporción de reales dentro de [p10, p90]; nominal 80 %",
    },
    "granularidad": {
        "semanal": "suma de T+1 a T+h por serie y corte; solo pares con los h días observados",
        "diaria": "cada día con real y pronóstico",
    },
    "segmentos": {
        "global": "todos los puntos",
        "evento": (
            "buen_fin, navidad, post_buen_fin o resto, con esa precedencia. En lo semanal, un corte entra a "
            "buen_fin o navidad si algún día de su horizonte cae en el evento; en lo diario cuenta el día. "
            "post_buen_fin son los cortes cuyos 7 días hasta el origen incluyen Buen Fin: los pronósticos "
            "que promedian esa ventana arrastran el pico a la semana siguiente. resto: las semanas normales"
        ),
        "formato": "store_format",
        "region": "region",
        "nse": "socioeconomic_level",
        "categoria": "category",
        "corte": "fecha de origen T",
        "horizonte": "h, solo en lo diario",
    },
    "columnas": {
        "n": "puntos evaluados: pares serie × corte completos (semanal) o días con real y pronóstico",
        "n_total": "puntos del segmento; n / n_total es la cobertura de la evaluación",
    },
}


class Backtest(NamedTuple):
    metricas: dict[str, Any]
    globales: dict[str, dict[str, Any]]


def _supuestos(horizonte: int) -> list[str]:
    return [
        "has_promotion se supone conocida de antemano en el horizonte: es un plan comercial.",
        "replenishment_signal fechada en T ya incluye las ventas de T+1: en la historia se pone en nulo.",
        "Los días sin units_sold, por nulo o por fila faltante, no se evalúan ni se imputan.",
        f"Las métricas semanales solo usan los pares serie × corte con los {horizonte} días observados.",
        "El factor del sistema actual es un escalar ajustado con todo el periodo (E1).",
        "AutoETS rellena los nulos de la historia solo para ajustarse; pronóstico e intervalo van en ≥ 0.",
    ]


def _verificar_malla(panel: pl.DataFrame) -> None:
    """Los desplazamientos por serie del backtest y de las features suponen la malla completa."""
    if panel.height != panel["date"].n_unique() * panel.select(SERIE).n_unique():
        raise ValueError("el panel no es la malla completa; corre `demanda calidad`")
    if panel.select(LLAVE).is_duplicated().any():
        raise ValueError("el panel repite la llave fecha × tienda × categoría")


def _cobertura(pronosticos: pl.DataFrame, horizonte: int) -> dict[str, Any]:
    """Cuántos puntos tienen real; no depende del modelo."""
    reales = pronosticos.filter(pl.col("modelo") == pronosticos["modelo"].first())
    pares = reales.group_by("origen", *SERIE).agg(
        ((pl.len() == horizonte) & pl.col("y").is_not_null().all()).alias("completo")
    )
    completos, observados = int(pares["completo"].sum()), reales["y"].is_not_null().sum()
    return {
        "semanal": {
            "pares_total": pares.height,
            "pares_completos": completos,
            "pct": round(100 * completos / pares.height, 2),
        },
        "diaria": {
            "dias_total": reales.height,
            "dias_observados": observados,
            "pct": round(100 * observados / reales.height, 2),
        },
    }


def _descripcion(modelo: Modelo) -> str:
    """La primera línea del docstring de la clase del modelo."""
    return next(iter((type(modelo).__doc__ or "").strip().splitlines()), "")


def ejecutar(
    panel_path: Path,
    reporte_path: Path,
    resumen_path: Path,
    salida_dir: Path,
    tracking_uri: str,
    mlruns_dir: Path,
    protocolo: Protocolo | None = None,
    n_jobs: int = 1,
    modelos: Sequence[str] | None = None,
    progreso: Callable[[str], None] | None = None,
) -> Backtest:
    """Corre el backtest y escribe `pronosticos.parquet` y `metricas.json` en `salida_dir`, más MLflow.

    Si `resumen_path` (el resumen del EDA) no existe, la conciliación reporta la referencia en nulo.
    """
    protocolo = protocolo or Protocolo()
    panel = pl.read_parquet(panel_path).sort([*SERIE, "date"])
    _verificar_malla(panel)
    reporte = json.loads(reporte_path.read_text())
    resumen = json.loads(resumen_path.read_text()) if resumen_path.exists() else None
    factor = reporte["fuga_replenishment_signal"]["factor"]

    elegidos = crear(Contexto(factor_senal=factor, n_jobs=n_jobs), modelos)
    resultado = bucle.correr(panel, elegidos, protocolo, progreso)
    tabla = segmentos.tabla(resultado, panel, protocolo, list(elegidos))
    origenes = protocolo.origenes(ultima_fecha(panel))

    metricas = {
        "fuente": {
            "panel_sha256": sha256(panel_path),
            "reporte_calidad_sha256": sha256(reporte_path),
            "resumen_eda_sha256": sha256(resumen_path) if resumen is not None else None,
            "datos_crudos_sha256": reporte.get("fuente", {}),
        },
        "protocolo": {
            "n_cortes": protocolo.n_cortes,
            "horizonte": protocolo.horizonte,
            "cuantiles": list(protocolo.cuantiles),
            "origenes": [str(o) for o in origenes],
            "factor_senal": factor,
            "modelos": list(elegidos),
            "supuestos": _supuestos(protocolo.horizonte),
        },
        "cobertura": _cobertura(resultado.pronosticos, protocolo.horizonte),
        "conciliacion_e2": conciliacion.conciliar(panel, factor, protocolo.horizonte, resumen),
        "segmentos": tabla.to_dicts(),
        "definiciones": {**DEFINICIONES, "modelos": {n: _descripcion(m) for n, m in elegidos.items()}},
    }

    salida_dir.mkdir(parents=True, exist_ok=True)
    orden_modelo = pl.col("modelo").cast(pl.Enum(list(elegidos)))
    resultado.pronosticos.sort(orden_modelo, "origen", *SERIE, "date").write_parquet(
        salida_dir / "pronosticos.parquet"
    )
    metricas_path = salida_dir / "metricas.json"
    metricas_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n")

    globales = segmentos.globales(tabla)
    parametros: Mapping[str, Any] = {
        "n_cortes": protocolo.n_cortes,
        "horizonte": protocolo.horizonte,
        "cuantiles": list(protocolo.cuantiles),
        "primer_origen": origenes[0],
        "ultimo_origen": origenes[-1],
        "factor_senal": factor,
    }
    seguimiento.registrar_runs(tracking_uri, mlruns_dir, parametros, globales, metricas_path)
    return Backtest(metricas, globales)
