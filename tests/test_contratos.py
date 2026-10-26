from collections.abc import Callable

import polars as pl
import pytest

from demanda.datos.contratos import ContratoRoto, Tablas, validar


def _en_primera_fila(columna: str, valor: object, dtype: pl.DataType | None = None) -> Callable:
    """Mutación: cambia el valor de `columna` en la primera fila."""

    def mutar(df: pl.DataFrame) -> pl.DataFrame:
        nuevo = pl.lit(valor, dtype=dtype or df.schema[columna])
        return df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(nuevo).otherwise(pl.col(columna)).alias(columna)
        )

    return mutar


def _transacciones(mutar: Callable) -> Callable[[Tablas], Tablas]:
    return lambda t: t._replace(transacciones=mutar(t.transacciones))


def _tiendas(mutar: Callable) -> Callable[[Tablas], Tablas]:
    return lambda t: t._replace(tiendas=mutar(t.tiendas))


def _calendario(mutar: Callable) -> Callable[[Tablas], Tablas]:
    return lambda t: t._replace(calendario=mutar(t.calendario))


def _duplicar_primera(df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([df, df.head(1)])


# (mutación, tabla, columna, fragmento de la regla que debe fallar)
VIOLACIONES = {
    "llave duplicada": (_transacciones(_duplicar_primera), "transacciones", "*", "uniqueness"),
    "categoría fuera de dominio": (
        _transacciones(_en_primera_fila("category", "Juguetes")),
        "transacciones",
        "category",
        "isin",
    ),
    "unidades negativas": (
        _transacciones(_en_primera_fila("units_sold", -1.0)),
        "transacciones",
        "units_sold",
        "greater_than_or_equal_to",
    ),
    "unidades no enteras": (
        _transacciones(_en_primera_fila("units_sold", 2.5)),
        "transacciones",
        "units_sold",
        "entero",
    ),
    "montos que no cuadran": (
        _transacciones(lambda df: df.with_columns(pl.col("amount_card") + 10)),
        "transacciones",
        "*",
        "montos",
    ),
    "más pagos que transacciones": (
        _transacciones(
            lambda df: df.with_columns(
                pl.col("total_transactions").cast(pl.Float64).alias("cash_transactions")
            )
        ),
        "transacciones",
        "*",
        "conteos",
    ),
    "efectivo nulo solo en el monto": (
        _transacciones(lambda df: df.with_columns(pl.lit(None, pl.Float64).alias("amount_cash"))),
        "transacciones",
        "*",
        "co-nulidad",
    ),
    "tienda nula": (
        _transacciones(_en_primera_fila("store_id", None, pl.String())),
        "transacciones",
        "store_id",
        "not_nullable",
    ),
    "columna no declarada": (
        _transacciones(lambda df: df.with_columns(pl.lit(1).alias("extra"))),
        "transacciones",
        "*",
        "column_in_schema",
    ),
    "columna faltante": (
        _transacciones(lambda df: df.drop("avg_ticket")),
        "transacciones",
        "*",
        "column_in_dataframe",
    ),
    "formato desconocido": (
        _tiendas(_en_primera_fila("store_format", "Mega")),
        "tiendas",
        "store_format",
        "isin",
    ),
    "tienda repetida en el catálogo": (_tiendas(_duplicar_primera), "tiendas", "store_id", "unique"),
    "día de la semana incorrecto": (
        _calendario(lambda df: df.with_columns((pl.col("day_of_week") + 1) % 7)),
        "calendario",
        "*",
        "day_of_week coincide con date",
    ),
    "hueco en el calendario": (
        _calendario(lambda df: df.filter(pl.int_range(pl.len()) != 10)),
        "calendario",
        "*",
        "fechas consecutivas",
    ),
    "festivo sin nombre": (
        _calendario(_en_primera_fila("holiday_name", None, pl.String())),
        "calendario",
        "*",
        "holiday_name solo en festivos",
    ),
    "tienda fuera del catálogo": (
        _tiendas(lambda df: df.slice(1)),
        "transacciones",
        "store_id",
        "existe en stores",
    ),
    "fecha fuera del calendario": (
        _calendario(lambda df: df.head(-1)),
        "transacciones",
        "date",
        "existe en calendar",
    ),
}


def test_datos_validos_cumplen_el_contrato(tablas):
    assert validar(tablas) is tablas


@pytest.mark.parametrize(("mutar", "tabla", "columna", "regla"), VIOLACIONES.values(), ids=VIOLACIONES.keys())
def test_cada_violacion_rompe_el_contrato(tablas, mutar, tabla, columna, regla):
    with pytest.raises(ContratoRoto) as error:
        validar(mutar(tablas))
    fallas = error.value.fallas.filter(
        (pl.col("tabla") == tabla)
        & (pl.col("columna") == columna)
        & pl.col("regla").str.contains(regla, literal=True)
    )
    assert fallas.height == 1, error.value.fallas


def test_reporta_todas_las_fallas_juntas(tablas):
    mutar = _transacciones(_en_primera_fila("category", "Juguetes"))
    roto = mutar(tablas)._replace(tiendas=_en_primera_fila("region", "Atlántida")(tablas.tiendas))
    with pytest.raises(ContratoRoto) as error:
        validar(roto)
    assert set(error.value.fallas["tabla"]) == {"transacciones", "tiendas"}
