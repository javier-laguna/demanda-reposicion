"""Panel tienda × categoría × día: las tres tablas unidas sobre la malla completa, con banderas de calidad.

La malla es completa a propósito: las filas que faltan en `transactions` aparecen con sus medidas en nulo y
`flag_fila_faltante`, así un `shift` por serie siempre corresponde a días de calendario. Nada se imputa.
"""

import polars as pl

from demanda.datos.contratos import LLAVE, SERIE, Tablas

COLUMNAS_CON_NULOS = ("cash_transactions", "amount_cash", "units_sold", "avg_ticket", "replenishment_signal")


def malla(transacciones: pl.DataFrame) -> pl.DataFrame:
    """Todas las combinaciones fecha × tienda × categoría entre la primera y la última fecha observadas."""
    fechas = transacciones.select(
        pl.date_range(pl.col("date").min(), pl.col("date").max(), "1d").alias("date")
    )
    return (
        fechas.join(transacciones.select("store_id").unique(), how="cross")
        .join(transacciones.select("category").unique(), how="cross")
        .select(LLAVE)
    )


def construir(tablas: Tablas) -> pl.DataFrame:
    t, tiendas, calendario = tablas
    medidas = [c for c in t.columns if c not in LLAVE]
    existe = ~pl.col("flag_fila_faltante")
    banderas = [
        (
            existe
            & pl.col("cash_transactions").is_not_null()
            & (pl.col("cash_transactions") + pl.col("card_transactions") < pl.col("total_transactions"))
        ).alias("flag_conteo_inconsistente"),
        *((existe & pl.col(c).is_null()).alias(f"flag_nulo_{c}") for c in COLUMNAS_CON_NULOS),
    ]
    return (
        malla(t)
        .join(t.with_columns(pl.lit(False).alias("flag_fila_faltante")), on=LLAVE, how="left")
        .with_columns(pl.col("flag_fila_faltante").fill_null(True))
        .with_columns(*banderas, pl.col("cash_transactions", "units_sold").cast(pl.Int64))
        .join(tiendas, on="store_id", how="left")
        .join(calendario, on="date", how="left")
        .select(
            *LLAVE,
            *medidas,
            "flag_fila_faltante",
            pl.col("^flag_(conteo|nulo).*$"),
            *(c for c in tiendas.columns if c != "store_id"),
            *(c for c in calendario.columns if c != "date"),
        )
        .sort([*SERIE, "date"])  # los joins de Polars no garantizan el orden
    )
