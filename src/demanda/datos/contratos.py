"""Contratos de datos (Pandera, backend Polars) para las tres tablas crudas y sus uniones.

Un contrato describe lo que debe cumplirse siempre: tipos, dominios, llaves e identidades contables. Si se
rompe, el archivo no es confiable y el pipeline se detiene. Los problemas conocidos de los datos (conteos
inconsistentes, nulos, filas faltantes) no rompen el contrato: se miden en `calidad` y se marcan en el panel.
"""

from collections.abc import Callable
from typing import NamedTuple

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData
from pandera.errors import SchemaErrors

CATEGORIAS = ("Abarrotes", "Bebidas", "Cuidado_Personal", "Electronica", "Hogar", "Ropa")
FORMATOS = ("Bodega", "Express", "Supercenter")
REGIONES = ("Centro", "Norte", "Occidente", "Oriente", "Sur")
NSE = ("A/B", "B", "C", "C+")
TEMPORADAS = ("Invierno", "Otoño", "Primavera", "Verano")
DIAS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

LLAVE = ("date", "store_id", "category")
SERIE = ("store_id", "category")
TOLERANCIA_MONTO = 0.01  # pesos
PATRON_TIENDA = r"^STR_\d{3}$"


class Tablas(NamedTuple):
    transacciones: pl.DataFrame
    tiendas: pl.DataFrame
    calendario: pl.DataFrame


class ContratoRoto(ValueError):
    """Los datos no cumplen el contrato. `fallas` resume qué regla falló, en qué tabla y cuántas veces."""

    def __init__(self, fallas: pl.DataFrame):
        self.fallas = fallas
        detalle = "\n".join(
            f"  {r['tabla']}.{r['columna']}: {r['regla']} ({r['casos']})" for r in fallas.to_dicts()
        )
        super().__init__(f"Los datos no cumplen el contrato:\n{detalle}")


def _regla(nombre: str, expr: Callable[[], pl.Expr]) -> pa.Check:
    """Check a nivel tabla: `expr` debe ser verdadera en cada fila."""

    def evaluar(data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(expr().alias(nombre))

    return pa.Check(evaluar, name=nombre)


def _entero(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key) == pl.col(data.key).round())


ENTERO = pa.Check(_entero, name="entero")


TRANSACCIONES = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Date),
        "store_id": pa.Column(pl.String, pa.Check.str_matches(PATRON_TIENDA)),
        "category": pa.Column(pl.String, pa.Check.isin(CATEGORIAS)),
        "total_transactions": pa.Column(pl.Int64, pa.Check.gt(0)),
        # El CSV escribe como float los conteos que tienen nulos; el contrato exige que sean enteros
        "cash_transactions": pa.Column(pl.Float64, [pa.Check.ge(0), ENTERO], nullable=True),
        "card_transactions": pa.Column(pl.Int64, pa.Check.ge(0)),
        "amount_total": pa.Column(pl.Float64, pa.Check.gt(0)),
        "amount_cash": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        "amount_card": pa.Column(pl.Float64, pa.Check.ge(0)),
        "units_sold": pa.Column(pl.Float64, [pa.Check.ge(0), ENTERO], nullable=True),
        "avg_ticket": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),
        "has_promotion": pa.Column(pl.Int64, pa.Check.isin([0, 1])),
        "replenishment_signal": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    },
    checks=[
        _regla(
            "montos: cash + card == total",
            lambda: (
                pl.col("amount_cash").is_null()
                | (
                    (pl.col("amount_cash") + pl.col("amount_card") - pl.col("amount_total")).abs()
                    <= TOLERANCIA_MONTO
                )
            ),
        ),
        _regla(
            "conteos: cash + card <= total",
            lambda: (
                pl.col("cash_transactions").is_null()
                | (pl.col("cash_transactions") + pl.col("card_transactions") <= pl.col("total_transactions"))
            ),
        ),
        _regla(
            "co-nulidad: cash_transactions y amount_cash",
            lambda: pl.col("cash_transactions").is_null() == pl.col("amount_cash").is_null(),
        ),
    ],
    unique=list(LLAVE),
    strict=True,
    name="transactions",
)

TIENDAS = pa.DataFrameSchema(
    {
        "store_id": pa.Column(pl.String, pa.Check.str_matches(PATRON_TIENDA), unique=True),
        "store_format": pa.Column(pl.String, pa.Check.isin(FORMATOS)),
        "region": pa.Column(pl.String, pa.Check.isin(REGIONES)),
        "size_sqm": pa.Column(pl.Int64, pa.Check.gt(0)),
        "num_checkouts": pa.Column(pl.Int64, pa.Check.gt(0)),
        "opening_year": pa.Column(pl.Int64, pa.Check.ge(1900)),
        "socioeconomic_level": pa.Column(pl.String, pa.Check.isin(NSE)),
        "has_pharmacy": pa.Column(pl.Boolean),
        "has_fuel_station": pa.Column(pl.Boolean),
    },
    strict=True,
    name="stores",
)

_DERIVADAS_DE_FECHA: dict[str, Callable[[], pl.Expr]] = {
    "day_of_week": lambda: pl.col("date").dt.weekday() - 1,  # 0 = lunes
    "day_name": lambda: pl.col("date").dt.strftime("%A"),
    "week_of_year": lambda: pl.col("date").dt.week(),  # ISO
    "month": lambda: pl.col("date").dt.month(),
    "year": lambda: pl.col("date").dt.year(),
    "quarter": lambda: pl.col("date").dt.quarter(),
    "is_weekend": lambda: pl.col("date").dt.weekday() >= 6,
}


def _coincide_con_fecha(columna: str, esperado: Callable[[], pl.Expr]) -> pa.Check:
    return _regla(f"{columna} coincide con date", lambda: pl.col(columna) == esperado())


CALENDARIO = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Date, unique=True),
        "day_of_week": pa.Column(pl.Int64, pa.Check.in_range(0, 6)),
        "day_name": pa.Column(pl.String, pa.Check.isin(DIAS)),
        "week_of_year": pa.Column(pl.Int64, pa.Check.in_range(1, 53)),
        "month": pa.Column(pl.Int64, pa.Check.in_range(1, 12)),
        "year": pa.Column(pl.Int64),
        "quarter": pa.Column(pl.Int64, pa.Check.in_range(1, 4)),
        "season": pa.Column(pl.String, pa.Check.isin(TEMPORADAS)),
        "is_holiday": pa.Column(pl.Boolean),
        "holiday_name": pa.Column(pl.String, nullable=True),
        "is_payday": pa.Column(pl.Boolean),
        "is_weekend": pa.Column(pl.Boolean),
        "is_navidad_season": pa.Column(pl.Boolean),
        "is_buen_fin": pa.Column(pl.Boolean),
        "is_semana_santa": pa.Column(pl.Boolean),
    },
    checks=[
        *(_coincide_con_fecha(columna, esperado) for columna, esperado in _DERIVADAS_DE_FECHA.items()),
        _regla(
            "holiday_name solo en festivos",
            lambda: pl.col("is_holiday") == pl.col("holiday_name").is_not_null(),
        ),
        _regla(
            "fechas consecutivas",
            lambda: pl.col("date").sort().diff().fill_null(pl.duration(days=1)) == pl.duration(days=1),
        ),
    ],
    strict=True,
    name="calendar",
)

ESQUEMAS = {"transacciones": TRANSACCIONES, "tiendas": TIENDAS, "calendario": CALENDARIO}


def _resumir(tabla: str, error: SchemaErrors) -> pl.DataFrame:
    casos = error.failure_cases
    if not isinstance(casos, pl.DataFrame):
        casos = pl.from_pandas(casos)
    return (
        casos.select(
            pl.lit(tabla).alias("tabla"),
            # En los checks de tabla Pandera pone el nombre del esquema como columna
            pl.when(pl.col("schema_context") == "DataFrameSchema")
            .then(pl.lit("*"))
            .otherwise(pl.col("column"))
            .alias("columna"),
            pl.col("check").alias("regla"),
        )
        .group_by("tabla", "columna", "regla", maintain_order=True)
        .len("casos")
    )


def _fallas_de_union(tablas: Tablas) -> pl.DataFrame:
    t, s, c = tablas
    sin_tienda = t.join(s, on="store_id", how="anti")["store_id"].unique()
    sin_fecha = t.join(c, on="date", how="anti")["date"].unique()
    filas = []
    if len(sin_tienda):
        filas.append(("transacciones", "store_id", "existe en stores", len(sin_tienda)))
    if len(sin_fecha):
        filas.append(("transacciones", "date", "existe en calendar", len(sin_fecha)))
    return pl.DataFrame(
        filas,
        schema={"tabla": pl.String, "columna": pl.String, "regla": pl.String, "casos": pl.UInt32},
        orient="row",
    )


def validar(tablas: Tablas) -> Tablas:
    """Valida las tres tablas y sus uniones. Junta todas las fallas antes de lanzar `ContratoRoto`."""
    fallas = []
    for (nombre, esquema), df in zip(ESQUEMAS.items(), tablas, strict=True):
        try:
            esquema.validate(df, lazy=True)
        except SchemaErrors as e:
            fallas.append(_resumir(nombre, e))
    # Las uniones solo tienen sentido si las llaves ya tienen el tipo correcto
    if not fallas:
        fallas.append(_fallas_de_union(tablas))
    resumen = pl.concat(fallas)
    if resumen.height:
        raise ContratoRoto(resumen)
    return tablas
