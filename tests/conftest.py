"""Datos sintéticos con la misma forma que los reales. Los tests nunca leen `data/raw/`."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from demanda.backtest.protocolo import CONOCIDAS, DE_LA_SENAL
from demanda.datos import panel
from demanda.datos.contratos import FORMATOS, NSE, REGIONES, SERIE, Tablas

INICIO = date(2023, 1, 1)


def generar(
    n_tiendas: int = 3,
    categorias: tuple[str, ...] = ("Abarrotes", "Bebidas"),
    dias: int = 56,
    ventana_senal: tuple[int, int] = (-5, 1),
    semilla: int = 0,
) -> Tablas:
    """Tablas válidas con los problemas conocidos de los datos reales, en proporciones parecidas.

    - Cada quinta fila tiene conteos inconsistentes (`cash + card < total`) y `avg_ticket` usa `cash + card`.
    - ~6 % de nulos simultáneos en `cash_transactions` y `amount_cash`; `avg_ticket` es nulo en parte de esos.
    - ~3 % de nulos en `units_sold` y la señal nula en las dos primeras fechas.
    - `replenishment_signal` = 0.53 × la media de `units_sold` de t + `desde` a t + `hasta` (`ventana_senal`).
    """
    rng = np.random.default_rng(semilla)
    fechas = pl.date_range(INICIO, INICIO + timedelta(days=dias - 1), "1d", eager=True)
    calendario = pl.DataFrame({"date": fechas}).with_columns(
        (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("day_of_week"),
        pl.col("date").dt.strftime("%A").alias("day_name"),
        pl.col("date").dt.week().cast(pl.Int64).alias("week_of_year"),
        pl.col("date").dt.month().cast(pl.Int64).alias("month"),
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
        pl.col("date").dt.quarter().cast(pl.Int64).alias("quarter"),
        pl.lit("Invierno").alias("season"),
        (pl.col("date") == INICIO).alias("is_holiday"),
        pl.when(pl.col("date") == INICIO).then(pl.lit("Año Nuevo")).alias("holiday_name"),
        (pl.col("date").dt.day() == 15).alias("is_payday"),
        (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
        pl.lit(False).alias("is_navidad_season"),
        pl.lit(False).alias("is_buen_fin"),
        pl.lit(False).alias("is_semana_santa"),
    )

    tiendas = pl.DataFrame(
        {
            "store_id": [f"STR_{i:03d}" for i in range(1, n_tiendas + 1)],
            "store_format": [FORMATOS[i % len(FORMATOS)] for i in range(n_tiendas)],
            "region": [REGIONES[i % len(REGIONES)] for i in range(n_tiendas)],
            "size_sqm": [5000 + 1000 * i for i in range(n_tiendas)],
            "num_checkouts": [10 + i for i in range(n_tiendas)],
            "opening_year": [2015] * n_tiendas,
            "socioeconomic_level": [NSE[i % len(NSE)] for i in range(n_tiendas)],
            "has_pharmacy": [i % 2 == 0 for i in range(n_tiendas)],
            "has_fuel_station": [False] * n_tiendas,
        }
    )

    # La señal puede usar días posteriores: se generan ventas de más y se recortan al final
    desde, hasta = ventana_senal
    extra = pl.date_range(INICIO, INICIO + timedelta(days=dias - 1 + max(hasta, 0)), "1d", eager=True)
    malla = (
        pl.DataFrame({"date": extra})
        .join(tiendas.select("store_id"), how="cross")
        .join(pl.DataFrame({"category": list(categorias)}), how="cross")
        .sort([*SERIE, "date"])
    )
    n = malla.height
    nivel = malla.select(
        pl.col("store_id").rank("dense") * 100 + pl.col("category").rank("dense") * 40
    ).to_series()
    fin_de_semana = malla.select(pl.col("date").dt.weekday() >= 6).to_series()
    unidades = np.round(nivel.to_numpy() * np.where(fin_de_semana, 1.5, 1.0) * rng.lognormal(0, 0.2, n))
    total = np.maximum(np.round(unidades / 2.5), 2).astype(np.int64)
    tarjeta = np.round(total * 0.6).astype(np.int64)
    efectivo = (total - tarjeta).astype(np.float64)
    efectivo[np.arange(n) % 5 == 0] -= 1  # conteos inconsistentes
    monto = np.round(total * 300 * rng.lognormal(0, 0.1, n), 2)
    monto_efectivo = np.round(monto * efectivo / (efectivo + tarjeta), 2)
    ticket = np.round(monto / (efectivo + tarjeta), 2)

    senal = malla.with_columns(pl.Series("u", unidades)).select(
        (pl.col("u").rolling_mean(hasta - desde + 1, min_samples=1).shift(-hasta).over(SERIE) * 0.53).round(2)
    )

    sin_efectivo = rng.random(n) < 0.06
    sin_ticket = sin_efectivo & (rng.random(n) < 0.4)
    sin_unidades = rng.random(n) < 0.03
    transacciones = malla.with_columns(
        pl.Series("total_transactions", total),
        pl.Series("cash_transactions", np.where(sin_efectivo, np.nan, efectivo)).fill_nan(None),
        pl.Series("card_transactions", tarjeta),
        pl.Series("amount_total", monto),
        pl.Series("amount_cash", np.where(sin_efectivo, np.nan, monto_efectivo)).fill_nan(None),
        pl.Series("amount_card", np.round(monto - monto_efectivo, 2)),
        pl.Series("units_sold", np.where(sin_unidades, np.nan, unidades)).fill_nan(None),
        pl.Series("avg_ticket", np.where(sin_ticket, np.nan, ticket)).fill_nan(None),
        pl.Series("has_promotion", (rng.random(n) < 0.2).astype(np.int64)),
        senal.to_series().alias("replenishment_signal"),
    ).with_columns(pl.when(pl.col("date") >= INICIO + timedelta(days=2)).then(pl.col("replenishment_signal")))
    return Tablas(transacciones.filter(pl.col("date").is_in(fechas.implode())), tiendas, calendario)


def escribir_csv(tablas: Tablas, carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    tablas.transacciones.write_csv(carpeta / "transactions.csv")
    tablas.tiendas.write_csv(carpeta / "stores.csv")
    tablas.calendario.write_csv(carpeta / "calendar.csv")
    return carpeta


@pytest.fixture
def tablas() -> Tablas:
    return generar()


@pytest.fixture
def generar_tablas():
    """La función `generar`, para los tests que necesitan otros parámetros."""
    return generar


@pytest.fixture
def carpeta_csv(tmp_path: Path):
    """Escribe unas tablas como CSV en una carpeta temporal y devuelve la carpeta."""
    return lambda tablas: escribir_csv(tablas, tmp_path / "raw")


def alterar_futuro(p: pl.DataFrame, origen: date) -> pl.DataFrame:
    """Cambia, después de `origen`, todas las columnas que no se conocen de antemano, y también la señal y su
    bandera fechadas en `origen`. Un pronóstico o una feature sin fugas no debe moverse."""
    despues = pl.col("date") > origen

    def alterar(columna: str) -> pl.Expr:
        nuevo = ~pl.col(columna) if p.schema[columna] == pl.Boolean else pl.col(columna).fill_null(0) * 3 + 7
        cuando = despues | (pl.col("date") == origen) if columna in DE_LA_SENAL else despues
        return pl.when(cuando).then(nuevo).otherwise(pl.col(columna)).alias(columna)

    return p.with_columns(alterar(c) for c in p.columns if c not in CONOCIDAS)


@pytest.fixture(scope="session")
def panel_sintetico() -> pl.DataFrame:
    """Panel de 70 días: alcanza para dos cortes semanales con ocho semanas de historia."""
    return panel.construir(generar(dias=70))


@pytest.fixture(scope="session")
def alterador():
    """La función `alterar_futuro`, para las pruebas de fugas."""
    return alterar_futuro
