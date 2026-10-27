"""Ajuste de hiperparámetros de LightGBM, solo con orígenes anteriores al backtest.

- Orígenes: los 8 jueves anteriores al primer corte del backtest (con los datos actuales, del 2023-09-07 al
  2023-10-26). Sus horizontes terminan en el primer origen del backtest, antes de su primer horizonte. Cada
  origen se entrena con su propia historia, con el mismo `separar` del backtest.
- Rejilla de 4 configuraciones, evaluada solo con P50 por costo y escogida por WAPE semanal sobre los pares
  completos de los 8 orígenes. Si empatan, gana la primera de la rejilla.
- El sistema actual en esos orígenes queda como referencia.

La ganadora se copia a mano como `gbm.AJUSTADOS`, así el backtest no depende de correr esto antes.
"""

import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from demanda.backtest import seguimiento
from demanda.backtest.bucle import validar_salida
from demanda.backtest.protocolo import Protocolo, separar, ultima_fecha
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256
from demanda.features.matriz import unidades
from demanda.modelos.baselines import SistemaActual
from demanda.modelos.gbm import Hiperparametros, LightGBMCuantiles, preparar
from demanda.seleccion.diebold_mariano import errores_semanales

N_ORIGENES = 8
REJILLA = tuple(
    Hiperparametros(num_leaves=hojas, learning_rate=tasa, n_estimators=arboles)
    for hojas in (31, 63)
    for tasa, arboles in ((0.05, 600), (0.1, 300))
)
REFERENCIA = "sistema_actual"
EXPERIMENTO = "ajuste_lightgbm"


def nombre(h: Hiperparametros) -> str:
    return f"hojas{h.num_leaves}_tasa{h.learning_rate}_arboles{h.n_estimators}"


def parametros_registrados(h: Hiperparametros, semilla: int = 0) -> dict[str, Any]:
    """Los parámetros con que LightGBM entrena la configuración, los mismos que usa el backtest. Sin los
    hilos, que dependen de la máquina y no cambian el resultado."""
    lgb = LightGBMCuantiles(factor=1.0, hiper=h, semilla=semilla).parametros_lgb(0.5)
    return {**{k: v for k, v in lgb.items() if k != "num_threads"}, "num_boost_round": h.n_estimators}


def antes_del_backtest(origenes: list[date], primero: date, horizonte: int) -> None:
    """Ningún horizonte de ajuste pasa del primer origen del backtest, así no toca días que se evalúan ahí."""
    if any(o + timedelta(days=horizonte) > primero for o in origenes):
        raise ValueError("el ajuste no puede usar días del backtest")


def origenes_de_ajuste(panel: pl.DataFrame, protocolo: Protocolo, n: int = N_ORIGENES) -> list[date]:
    """Los n orígenes semanales anteriores al primer corte del backtest, en orden cronológico."""
    primero = protocolo.origenes(ultima_fecha(panel))[0]
    origenes = [primero - timedelta(days=protocolo.horizonte * k) for k in range(n, 0, -1)]
    antes_del_backtest(origenes, primero, protocolo.horizonte)
    return origenes


def _wape(errores: pl.DataFrame, por: list[str]) -> pl.DataFrame:
    return errores.group_by(por).agg(
        (100 * pl.col("error").abs().sum() / pl.col("y").sum()).round(4).alias("wape_semanal_pct"),
        pl.len().alias("pares"),
    )


def ejecutar(
    panel_path: Path,
    reporte_path: Path,
    salida_path: Path,
    tracking_uri: str,
    mlruns_dir: Path,
    protocolo: Protocolo | None = None,
    semilla: int = 0,
    n_origenes: int = N_ORIGENES,
    rejilla: Sequence[Hiperparametros] = REJILLA,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evalúa la rejilla en los orígenes de ajuste y escribe `ajuste_lightgbm.json`, más un run de MLflow por
    configuración. Las matrices se arman una vez por origen, y las configuraciones se entrenan a la vez con
    los núcleos repartidos."""
    protocolo = protocolo or Protocolo()
    panel = pl.read_parquet(panel_path).sort([*SERIE, "date"])
    factor = json.loads(reporte_path.read_text())["fuga_replenishment_signal"]["factor"]
    origenes = origenes_de_ajuste(panel, protocolo, n_origenes)
    hilos = max(1, (os.cpu_count() or 1) // len(rejilla))
    modelos = {
        nombre(h): LightGBMCuantiles(factor, hiper=h, cuantiles=(0.5,), semilla=semilla, hilos=hilos)
        for h in rejilla
    }
    referencia = SistemaActual(factor)

    pronosticos = []
    for origen in origenes:
        corte = separar(panel, origen, protocolo.horizonte)
        historia, futuro = corte.historia, corte.futuro
        prep = preparar(historia, futuro, factor, LightGBMCuantiles.paso)
        with ThreadPoolExecutor(len(modelos)) as grupo:
            en_paralelo = grupo.map(lambda m, p=prep, f=futuro: m.predecir(p, f), modelos.values())
            preds = dict(zip(modelos, en_paralelo, strict=True))
        preds[REFERENCIA] = referencia.pronosticar(historia, futuro)
        for n, pred in preds.items():
            pronosticos.append(
                validar_salida(pred, futuro, n).select(
                    pl.lit(n).alias("modelo"), pl.lit(origen).alias("origen"), *SERIE, "date", "h", "yhat"
                )
            )
        if progreso:
            progreso(f"origen de ajuste {origen} listo")

    reales = panel.select(*SERIE, "date", unidades().alias("y"))
    errores = errores_semanales(
        pl.concat(pronosticos).join(reales, on=[*SERIE, "date"], how="left"), protocolo.horizonte
    )
    globales = {m: (w, n) for m, w, n in _wape(errores, ["modelo"]).iter_rows()}
    por_origen = _wape(errores, ["modelo", "origen"]).sort("origen")

    def detalle(m: str) -> dict[str, Any]:
        del_modelo = por_origen.filter(pl.col("modelo") == m)
        return {
            "wape_semanal_pct": globales[m][0],
            "pares_completos": globales[m][1],
            "por_origen": {str(o): w for o, w in del_modelo.select("origen", "wape_semanal_pct").iter_rows()},
        }

    evaluadas: list[dict[str, Any]] = [
        {
            "nombre": nombre(h),
            **asdict(h),
            "parametros_lgb": parametros_registrados(h, semilla),
            **detalle(nombre(h)),
        }
        for h in rejilla
    ]
    ganadora = min(
        evaluadas, key=lambda r: r["wape_semanal_pct"]
    )  # min es estable: el empate va a la primera
    resultado = {
        "fuente": {"panel_sha256": sha256(panel_path), "reporte_calidad_sha256": sha256(reporte_path)},
        "protocolo": {
            "origenes": [str(o) for o in origenes],
            "primer_origen_backtest": str(protocolo.origenes(ultima_fecha(panel))[0]),
            "horizonte": protocolo.horizonte,
            "cuantil": 0.5,
            "paso_dias": LightGBMCuantiles.paso,
            "semilla": semilla,
            "criterio": "menor WAPE semanal sobre los pares completos de los orígenes; empate, la primera",
        },
        "referencia": {"modelo": REFERENCIA, **detalle(REFERENCIA)},
        "rejilla": evaluadas,
        "ganadora": ganadora,
    }
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    salida_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n")

    seguimiento.registrar_runs(
        tracking_uri,
        mlruns_dir,
        {"origenes": f"{origenes[0]}..{origenes[-1]}", "cuantil": 0.5, "semilla": semilla},
        {nombre(h): {"semanal_wape_pct": globales[nombre(h)][0]} for h in rejilla},
        salida_path,
        por_run={nombre(h): asdict(h) for h in rejilla},
        experimento=EXPERIMENTO,
    )
    return resultado
