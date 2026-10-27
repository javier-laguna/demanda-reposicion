"""Pronósticos del campeón y del retador en los orígenes previos al backtest, para calibrar E5.

- Orígenes: los de `ajuste.origenes_de_ajuste`, los 8 jueves previos al primer corte del backtest (con los
  datos actuales, del 2023-09-07 al 2023-10-26). Sus horizontes terminan en el primer origen del backtest.
- Modelos: el campeón y el retador de `seleccion.json`, con el mismo `separar` y el mismo `bucle` del
  backtest. LightGBM corre con sus tres cuantiles: el orden de los cuantiles puede mover `yhat`, y así es el
  mismo modelo.

La salida, `calibracion.parquet`, tiene las columnas de `pronosticos.parquet`. Con ella E5 mide los errores
semanales de antes de cada semana que decide: los cuantiles de servicio y la banda de desempeño del monitor.
Al lado, `calibracion.json` registra su fuente con hashes: `decision` y `drift` se niegan a usar una
calibración hecha con otro panel o con otra selección.
"""

import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import polars as pl

from demanda.backtest import bucle
from demanda.backtest.etapa import cerrar_modelos, verificar_malla
from demanda.backtest.protocolo import Protocolo
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256
from demanda.modelos import Contexto, crear
from demanda.modelos.ajuste import N_ORIGENES, antes_del_backtest, origenes_de_ajuste
from demanda.seleccion.campeon import campeon_y_retador


def registro(ruta: Path) -> Path:
    """El registro de la fuente, al lado del parquet."""
    return ruta.with_suffix(".json")


def leer(
    ruta: Path,
    panel_path: Path,
    seleccion_path: Path,
    modelos: Sequence[str],
    primer_corte: date,
    horizonte: int,
) -> pl.DataFrame:
    """Lee `calibracion.parquet` y verifica que salga de este panel y de esta selección (por los hashes de
    `calibracion.json`), que sea de estos modelos y que no toque días del backtest."""
    if not registro(ruta).exists():
        raise FileNotFoundError(f"no existe {registro(ruta)}; corre `demanda calibracion`")
    fuente = json.loads(registro(ruta).read_text())["fuente"]
    actuales = {"calibracion": ruta, "panel": panel_path, "seleccion": seleccion_path}
    distintos = [p.name for k, p in actuales.items() if fuente.get(f"{k}_sha256") != sha256(p)]
    if distintos:
        raise ValueError(
            f"{registro(ruta).name} no coincide con {', '.join(distintos)}: la calibración es de otra "
            "corrida; corre `demanda calibracion`"
        )
    calibracion = pl.read_parquet(ruta)
    if set(calibracion["modelo"].unique().to_list()) != set(modelos):
        raise ValueError(
            f"calibracion.parquet no es de {' y '.join(modelos)}, los de seleccion.json; "
            "corre `demanda calibracion`"
        )
    antes_del_backtest(calibracion["origen"].unique().to_list(), primer_corte, horizonte)
    return calibracion


def ejecutar(
    panel_path: Path,
    reporte_path: Path,
    seleccion_path: Path,
    salida_path: Path,
    protocolo: Protocolo | None = None,
    n_jobs: int = 1,
    n_origenes: int = N_ORIGENES,
    progreso: Callable[[str], None] | None = None,
) -> pl.DataFrame:
    """Corre el campeón y el retador en los orígenes de calibración y escribe `calibracion.parquet` y su
    registro, `calibracion.json`."""
    protocolo = protocolo or Protocolo()
    panel = pl.read_parquet(panel_path).sort([*SERIE, "date"])
    verificar_malla(panel)
    factor = json.loads(reporte_path.read_text())["fuga_replenishment_signal"]["factor"]
    campeon, retador = campeon_y_retador(json.loads(seleccion_path.read_text()))
    origenes = origenes_de_ajuste(panel, protocolo, n_origenes)

    elegidos = crear(Contexto(factor_senal=factor, n_jobs=n_jobs), [campeon, retador])
    try:
        resultado = bucle.correr(panel, elegidos, protocolo, progreso, origenes=origenes)
    finally:
        cerrar_modelos(elegidos)

    orden_modelo = pl.col("modelo").cast(pl.Enum(list(elegidos)))
    pronosticos = resultado.pronosticos.sort(orden_modelo, "origen", *SERIE, "date")
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    pronosticos.write_parquet(salida_path)
    fuente = {
        "calibracion_sha256": sha256(salida_path),
        "panel_sha256": sha256(panel_path),
        "reporte_calidad_sha256": sha256(reporte_path),
        "seleccion_sha256": sha256(seleccion_path),
    }
    contenido = {"fuente": fuente, "modelos": list(elegidos), "origenes": [str(o) for o in origenes]}
    registro(salida_path).write_text(json.dumps(contenido, indent=2, ensure_ascii=False) + "\n")
    return pronosticos
