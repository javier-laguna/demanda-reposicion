"""Modelos de pronóstico. Importar el paquete llena el registro, en el orden en que se reportan."""

# El orden de importación es el del registro: los baselines, luego LightGBM y al final Chronos-2
from demanda.modelos import baselines, ets, gbm

# isort: split
from demanda.modelos import chronos2
from demanda.modelos.registro import INTERVALO, REGISTRO, SALIDA, Contexto, Modelo, crear, registrar

__all__ = [
    "INTERVALO",
    "REGISTRO",
    "SALIDA",
    "Contexto",
    "Modelo",
    "baselines",
    "chronos2",
    "crear",
    "ets",
    "gbm",
    "registrar",
]
