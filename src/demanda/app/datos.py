"""Lectura de los artefactos para el tablero, con las rutas de `Config` y la caché de Streamlit.

- Cada lectura se cachea por ruta y fecha de modificación: si se vuelve a correr una etapa, la siguiente
  interacción ya lee el artefacto nuevo.
- La config se crea en cada corrida del script y no con `get_config`, así un cambio de `ARTIFACTS_DIR` (las
  pruebas lo hacen) no exige reiniciar el proceso.
- Si falta un artefacto, la lectura avisa en la página qué comando `make` lo genera y devuelve `None`; la
  página sigue sin esa sección.
"""

import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from demanda.config import Config
from demanda.mcp_server.herramientas import COMANDOS as COMANDOS_MCP

# Cada artefacto, relativo a `artifacts_dir`, con el comando que lo escribe
COMANDOS = {**COMANDOS_MCP, "insights/insights.json": "make insights"}
PRONOSTICOS = "backtest/pronosticos.parquet"
PANEL = "interim/panel.parquet"  # relativo a `data_dir`; lo escribe `make calidad`
# Los atributos de la tienda en el panel: del panel solo se leen estas columnas
TIENDA = (
    "store_id",
    "store_format",
    "region",
    "socioeconomic_level",
    "size_sqm",
    "num_checkouts",
    "opening_year",
    "has_pharmacy",
    "has_fuel_station",
)

# El pipeline en orden: comando, qué hace y lo que escribe. Las rutas empiezan con la carpeta de `Config`
# a la que pertenecen (`data`, `artifacts` o `reports`).
FLUJO: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("make setup", "ambiente con uv y hooks de pre-commit", ()),
    (
        "make data DATA_SRC=…",
        "copia los 3 CSV y verifica sus hashes",
        ("data/raw/transactions.csv", "data/raw/stores.csv", "data/raw/calendar.csv"),
    ),
    (
        "make calidad",
        "contratos, panel completo y reporte de calidad",
        ("data/interim/panel.parquet", "artifacts/calidad/reporte.json"),
    ),
    ("make eda", "notebook de EDA (marimo)", ("reports/eda.html", "artifacts/eda/resumen.json")),
    (
        "make backtest",
        "17 cortes × 6 modelos, con MLflow (~19 min)",
        ("artifacts/backtest/pronosticos.parquet", "artifacts/backtest/metricas.json"),
    ),
    ("make seleccion", "Diebold-Mariano y regla de campeón", ("artifacts/modelos/seleccion.json",)),
    (
        "make calibracion",
        "campeón y retador en los 8 orígenes previos al backtest",
        ("artifacts/backtest/calibracion.parquet",),
    ),
    ("make decision", "newsvendor: políticas, impacto y sensibilidad", ("artifacts/decision/impacto.json",)),
    ("make drift", "monitor semana a semana y drift inyectado", ("artifacts/drift/monitor.json",)),
    ("make causal", "experimento de promociones", ("artifacts/causal/promociones.json",)),
    ("make insights", "agentes, caché o plantillas", ("artifacts/insights/insights.json",)),
    ("make app", "este tablero", ()),
)
# El sello de tiempo con que archify nombra sus carpetas: `architecture-solucion-20261027-005059`
SELLO_ARCHIFY = re.compile(r"(\d{8}-\d{6})$")


def config() -> Config:
    return Config()


def avisar(ruta: str, comando: str) -> None:
    st.warning(f"Falta `{ruta}`. Genéralo con `{comando}`.", icon=":material/warning:")


@st.cache_data(show_spinner=False)
def _json(ruta: str, modificado: float) -> dict[str, Any]:
    return json.loads(Path(ruta).read_text())


def cargar(*relativas: str) -> list[dict[str, Any]] | None:
    """Los artefactos JSON pedidos, en orden, o `None` si falta alguno: por cada faltante, un aviso."""
    artefactos = config().artifacts_dir
    faltan = [r for r in relativas if not (artefactos / r).exists()]
    for r in faltan:
        avisar(f"{artefactos.name}/{r}", COMANDOS[r])
    if faltan:
        return None
    return [_json(str(artefactos / r), (artefactos / r).stat().st_mtime) for r in relativas]


def artefacto(relativa: str) -> dict[str, Any] | None:
    """Un artefacto JSON, o `None` con su aviso si falta."""
    leidos = cargar(relativa)
    return None if leidos is None else leidos[0]


def _pronosticos() -> Path | None:
    artefactos = config().artifacts_dir
    ruta = artefactos / PRONOSTICOS
    if not ruta.exists():
        avisar(f"{artefactos.name}/{PRONOSTICOS}", COMANDOS[PRONOSTICOS])
        return None
    return ruta


@st.cache_data(show_spinner=False)
def _catalogo(ruta: str, modificado: float) -> dict[str, list[str]]:
    catalogo = (
        pl.scan_parquet(ruta)
        .select(
            pl.col("store_id").unique().sort().implode(),
            pl.col("category").unique().sort().implode(),
            pl.col("origen").unique().sort().cast(pl.String).implode(),
        )
        .collect()
    )
    return {
        nombre: catalogo[c][0].to_list()
        for nombre, c in zip(("tiendas", "categorias", "origenes"), catalogo.columns, strict=True)
    }


def catalogo_pronosticos() -> dict[str, list[str]] | None:
    """Las tiendas, categorías y cortes (como texto) de `pronosticos.parquet`, o `None` con su aviso."""
    ruta = _pronosticos()
    return None if ruta is None else _catalogo(str(ruta), ruta.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _serie(
    ruta: str, modificado: float, store_id: str, category: str, modelos: tuple[str, ...]
) -> pl.DataFrame:
    return (
        pl.scan_parquet(ruta)
        .filter(
            pl.col("store_id") == store_id,
            pl.col("category") == category,
            pl.col("modelo").is_in(list(modelos)),
        )
        .sort("modelo", "origen", "h")
        .collect()
    )


def serie(store_id: str, category: str, modelos: tuple[str, ...]) -> pl.DataFrame | None:
    """Los pronósticos de una tienda × categoría en todos los cortes, solo de `modelos`."""
    ruta = _pronosticos()
    return None if ruta is None else _serie(str(ruta), ruta.stat().st_mtime, store_id, category, modelos)


@st.cache_data(show_spinner=False)
def _tiendas(ruta: str, modificado: float) -> pl.DataFrame:
    # Solo las columnas de la tienda: no se carga el panel completo
    return pl.scan_parquet(ruta).select(TIENDA).unique().sort("store_id").collect()


def tiendas() -> pl.DataFrame | None:
    """El catálogo de tiendas (formato, región, NSE, tamaño…) desde el panel, o `None` con su aviso."""
    datos = config().data_dir
    ruta = datos / PANEL
    if not ruta.exists():
        avisar(f"{datos.name}/{PANEL}", "make calidad")
        return None
    return _tiendas(str(ruta), ruta.stat().st_mtime)


def diagrama_mas_reciente(carpeta: Path) -> Path | None:
    """El HTML de la carpeta más reciente de `carpeta`: la de mayor sello `AAAAMMDD-HHMMSS` al final del
    nombre, no la de fecha de modificación más reciente, que cambia al clonar. Solo cuentan las carpetas con
    un HTML; si tienen varios, el primero en orden alfabético."""
    if not carpeta.is_dir():
        return None
    candidatas = []
    for sub in carpeta.iterdir():
        htmls = sorted(sub.glob("*.html")) if sub.is_dir() else []
        if htmls:
            sello = SELLO_ARCHIFY.search(sub.name)
            candidatas.append(((sello.group(1) if sello else "", sub.name), htmls[0]))
    return max(candidatas)[1] if candidatas else None


@st.cache_data(show_spinner=False)
def _texto(ruta: str, modificado: float) -> str:
    return Path(ruta).read_text(encoding="utf-8")


def diagrama() -> tuple[Path, str] | None:
    """La ruta y el HTML del diagrama más reciente, o `None` con su aviso."""
    carpeta = config().archify_dir
    ruta = diagrama_mas_reciente(carpeta)
    if ruta is None:
        st.warning(
            f"Falta el diagrama en `{carpeta.name}/`. Lo genera la skill `archify` de Claude Code; no lo "
            "escribe ningún `make`.",
            icon=":material/warning:",
        )
        return None
    return ruta, _texto(str(ruta), ruta.stat().st_mtime)


def estado_flujo(cfg: Config) -> list[dict[str, str]]:
    """Cada paso de `FLUJO` con sus salidas y si ya existen todas."""
    raices = {"data": cfg.data_dir, "artifacts": cfg.artifacts_dir, "reports": cfg.reports_dir}
    filas = []
    for comando, que_hace, salidas in FLUJO:
        rutas = [raices[s.split("/", 1)[0]] / s.split("/", 1)[1] for s in salidas]
        estado = "—" if not rutas else ("listo" if all(r.exists() for r in rutas) else "falta")
        filas.append(
            {"comando": comando, "qué hace": que_hace, "escribe": ", ".join(salidas), "estado": estado}
        )
    return filas
