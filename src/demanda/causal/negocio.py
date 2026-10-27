"""La traducción a negocio: unidades incrementales, utilidad con descuento y la regla de «se paga».

- Unidades incrementales de la categoría c: Σ y · (1 − e^(−β_c)) sobre los días con promoción, y su promedio
  por día-promoción (tienda × categoría × día). Es creciente en β, así el IC95 sale de evaluar en los extremos
  del IC de β_c.
- Precio de lista p_c: Σ `amount_total` / Σ `units_sold` en los días sin promoción.
- Margen m_c: el margen bruto de `decision/supuestos.py`, el mismo del newsvendor. El descuento d es un
  parámetro; si el IC95 del efecto sobre el precio por unidad queda abajo de cero, se agrega el implícito
  d̂ = 1 − e^(β_precio).
- Utilidad por día-promoción, con el contrafactual y₀ = y · e^(−β) y el lift L = e^β:
  Δπ = p · y₀ · [(m − d) · L − m]. También es creciente en β.
- Lift de equilibrio: L* − 1 = d / (m − d); con d ≥ m la promoción nunca se paga.
- Regla por categoría y descuento: «se paga» si el extremo inferior del IC95 del lift supera el de
  equilibrio, «no se paga» si el superior queda abajo e «indeterminado» en otro caso.
- Valor máximo de una promoción sin descuento: m · p · (unidades incrementales por día-promoción). Es lo más
  que podría costar una promoción que no baja el precio (exhibición, folleto) para pagarse.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import polars as pl

from demanda.decision.supuestos import COSTOS, Costo

PROMO = "has_promotion"
DESCUENTOS = (0.05, 0.10, 0.15, 0.20)
SE_PAGA, NO_SE_PAGA, INDETERMINADO = "se_paga", "no_se_paga", "indeterminado"
VEREDICTOS = (SE_PAGA, NO_SE_PAGA, INDETERMINADO)


@dataclass(frozen=True)
class SupuestosNegocio:
    """Los márgenes del newsvendor y la rejilla de descuentos, que los datos no traen."""

    costos: MappingProxyType[str, Costo] = field(default_factory=lambda: COSTOS)
    descuentos: tuple[float, ...] = DESCUENTOS

    def margen(self, categoria: str) -> float:
        return self.costos[categoria].margen

    def a_json(self) -> dict[str, Any]:
        return {
            "margen_bruto": {c: costo.margen for c, costo in self.costos.items()},
            "origen_margen": "decision/supuestos.py, el mismo del newsvendor (E5)",
            "descuentos": list(self.descuentos),
        }


def lift_equilibrio(margen: float, descuento: float) -> float | None:
    """L* − 1 = d / (m − d), el lift que deja la utilidad igual. Con d ≥ m no existe: None."""
    return None if descuento >= margen else descuento / (margen - descuento)


def utilidad(precio: float, y0: float, lift: float, margen: float, descuento: float) -> float:
    """Δπ = p · y₀ · [(m − d) · L − m], con `lift` = L = e^β."""
    return precio * y0 * ((margen - descuento) * lift - margen)


def veredicto(lift_inf: float, lift_sup: float, margen: float, descuento: float) -> str:
    """La regla con el IC95 del lift (L − 1) contra el lift de equilibrio."""
    equilibrio = lift_equilibrio(margen, descuento)
    if equilibrio is None:
        return NO_SE_PAGA
    if lift_inf > equilibrio:
        return SE_PAGA
    return NO_SE_PAGA if lift_sup < equilibrio else INDETERMINADO


def por_categoria(m: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Por categoría: los días con promoción, sus unidades y el precio de lista de los días sin promoción."""
    tabla = (
        m.group_by("category")
        .agg(
            (pl.col(PROMO) == 1).sum().alias("dias"),
            pl.col("units_sold").filter(pl.col(PROMO) == 1).sum().cast(pl.Float64).alias("unidades"),
            (
                pl.col("amount_total").filter(pl.col(PROMO) == 0).sum()
                / pl.col("units_sold").filter(pl.col(PROMO) == 0).sum()
            ).alias("precio"),
        )
        .sort("category")
    )
    return {
        f["category"]: {k: float(f[k]) for k in ("dias", "unidades", "precio")}
        for f in tabla.iter_rows(named=True)
    }


def _en(b: float, unidades: float, dias: float, precio: float, margen: float) -> dict[str, float]:
    """Lo que depende de β en una categoría."""
    incrementales = unidades * (1 - math.exp(-b))
    return {
        "lift": math.expm1(b),
        "incrementales": incrementales,
        "por_dia": incrementales / dias,
        "valor_maximo": margen * precio * incrementales / dias,
        "y0": unidades / dias * math.exp(-b),
    }


def traducir(
    m: pl.DataFrame,
    efectos: Mapping[str, Mapping[str, Any]],
    precio_unitario: Mapping[str, Any],
    s: SupuestosNegocio,
) -> dict[str, Any]:
    """La traducción por categoría. `efectos` son los de la heterogeneidad por categoría y
    `precio_unitario`, el canal del precio por unidad, los dos con `beta` e `ic95`."""
    implicito = precio_unitario["ic95"][1] < 0
    descuentos = [(d, "rejilla") for d in s.descuentos]
    if implicito:
        descuentos.append((-math.expm1(precio_unitario["beta"]), "implicito"))
    datos = por_categoria(m)
    salida: dict[str, Any] = {}
    for categoria, base in datos.items():
        margen = s.margen(categoria)
        b, (inf, sup) = efectos[categoria]["beta"], efectos[categoria]["ic95"]
        punto, bajo, alto = (
            _en(x, base["unidades"], base["dias"], base["precio"], margen) for x in (b, inf, sup)
        )
        filas = []
        for d, origen in descuentos:
            equilibrio = lift_equilibrio(margen, d)
            delta = [utilidad(base["precio"], e["y0"], 1 + e["lift"], margen, d) for e in (punto, bajo, alto)]
            filas.append(
                {
                    "descuento": d,
                    "origen": origen,
                    "lift_equilibrio_pct": None if equilibrio is None else 100 * equilibrio,
                    "utilidad_por_dia_promocion": delta[0],
                    "utilidad_por_dia_promocion_ic95": delta[1:],
                    "utilidad_total": base["dias"] * delta[0],
                    "veredicto": veredicto(bajo["lift"], alto["lift"], margen, d),
                }
            )
        salida[categoria] = {
            "dias_promocion": int(base["dias"]),
            "precio_lista": base["precio"],
            "margen_bruto": margen,
            "lift_pct": 100 * punto["lift"],
            "lift_ic95_pct": [100 * bajo["lift"], 100 * alto["lift"]],
            "unidades_incrementales": punto["incrementales"],
            "unidades_incrementales_ic95": [bajo["incrementales"], alto["incrementales"]],
            "incrementales_por_dia_promocion": punto["por_dia"],
            "incrementales_por_dia_promocion_ic95": [bajo["por_dia"], alto["por_dia"]],
            "valor_maximo_sin_descuento_por_dia": punto["valor_maximo"],
            "valor_maximo_sin_descuento_por_dia_ic95": [bajo["valor_maximo"], alto["valor_maximo"]],
            "por_descuento": filas,
        }
    resumen = []
    for i, (d, origen) in enumerate(descuentos):
        por_veredicto = {
            v: [c for c in salida if salida[c]["por_descuento"][i]["veredicto"] == v] for v in VEREDICTOS
        }
        resumen.append({"descuento": d, "origen": origen, **por_veredicto})
    return {
        "descuento_implicito": {
            "aplica": implicito,
            "descuento": -math.expm1(precio_unitario["beta"]) if implicito else None,
            "regla": "se agrega si el IC95 del efecto sobre el precio por unidad queda abajo de cero",
        },
        "por_categoria": salida,
        "por_descuento": resumen,
    }
