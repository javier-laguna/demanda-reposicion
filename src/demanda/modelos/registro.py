"""Registro de modelos por nombre, con una interfaz común: `pronosticar(historia, futuro)`.

El bucle del backtest solo conoce esta interfaz. Un modelo nuevo se registra con `registrar` y queda cubierto
por el backtest y por la prueba de fugas sin tocar ninguno de los dos.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import polars as pl

SALIDA = ("store_id", "category", "date", "h", "yhat")
INTERVALO = ("p10", "p90")


class Modelo(Protocol):
    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        """Una fila por fila de `futuro` con `SALIDA` y, si el modelo da intervalo, `INTERVALO`.

        `historia` es el panel hasta el origen T, con la señal de T en nulo; `futuro` trae de T+1 a T+h solo
        lo que se conoce de antemano. El modelo no recibe nada más.
        """
        ...


@dataclass(frozen=True)
class Contexto:
    """Lo que un modelo puede necesitar además del corte."""

    factor_senal: float
    n_jobs: int = 1


Fabrica = Callable[[Contexto], Modelo]
REGISTRO: dict[str, Fabrica] = {}


def registrar(nombre: str, fabrica: Fabrica) -> None:
    if nombre in REGISTRO:
        raise ValueError(f"ya hay un modelo registrado como {nombre}")
    REGISTRO[nombre] = fabrica


def crear(contexto: Contexto, nombres: Iterable[str] | None = None) -> dict[str, Modelo]:
    """Los modelos pedidos (todos, si `nombres` es None), en el orden del registro."""
    pedidos = list(REGISTRO) if nombres is None else list(nombres)
    desconocidos = [n for n in pedidos if n not in REGISTRO]
    if desconocidos:
        raise ValueError(f"modelos no registrados: {', '.join(desconocidos)}; hay: {', '.join(REGISTRO)}")
    return {n: REGISTRO[n](contexto) for n in REGISTRO if n in pedidos}
