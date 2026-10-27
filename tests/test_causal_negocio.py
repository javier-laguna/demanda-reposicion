"""La traducción a negocio contra casos calculados a mano: utilidad, lift de equilibrio, regla y unidades."""

import math
from types import MappingProxyType

import polars as pl
import pytest

from demanda.causal import negocio
from demanda.causal.negocio import NO_SE_PAGA, SE_PAGA, SupuestosNegocio
from demanda.decision.supuestos import COSTOS, Costo


def test_utilidad_y_lift_de_equilibrio_a_mano():
    # p = 10, y₀ = 100, L = 1.1, m = 0.3, d = 0.1: Δπ = 10 · 100 · (0.2 · 1.1 − 0.3) = −80
    assert negocio.utilidad(10, 100, 1.1, 0.3, 0.1) == pytest.approx(-80)
    # L* − 1 = 0.1 / (0.3 − 0.1) = 0.5, y ahí la utilidad es cero
    assert negocio.lift_equilibrio(0.3, 0.1) == pytest.approx(0.5)
    assert negocio.utilidad(10, 100, 1.5, 0.3, 0.1) == pytest.approx(0, abs=1e-12)
    # Sin descuento cualquier lift positivo se paga
    assert negocio.lift_equilibrio(0.3, 0.0) == 0
    # Con d ≥ m no hay lift que alcance: la utilidad es negativa siempre
    assert negocio.lift_equilibrio(0.3, 0.3) is None and negocio.lift_equilibrio(0.12, 0.15) is None
    assert negocio.utilidad(10, 100, 5.0, 0.12, 0.15) < 0


def test_la_regla_con_el_ic_del_lift():
    assert negocio.veredicto(0.6, 0.8, 0.3, 0.1) == SE_PAGA  # todo el IC arriba de 0.5
    assert negocio.veredicto(0.1, 0.4, 0.3, 0.1) == NO_SE_PAGA  # todo abajo
    assert negocio.veredicto(0.4, 0.6, 0.3, 0.1) == negocio.INDETERMINADO
    assert negocio.veredicto(5.0, 10.0, 0.3, 0.3) == NO_SE_PAGA  # d ≥ m


def _datos() -> pl.DataFrame:
    """Dos categorías: en A, 2 días con promoción (300 unidades) y un precio sin promoción de 10."""
    return pl.DataFrame(
        {
            "category": ["A"] * 4 + ["B"] * 2,
            "has_promotion": [1, 1, 0, 0, 1, 0],
            "units_sold": [100, 200, 100, 300, 50, 50],
            "amount_total": [900.0, 1800.0, 1000.0, 3000.0, 400.0, 1000.0],
        }
    )


def test_traducir_a_mano():
    s = SupuestosNegocio(
        costos=MappingProxyType({"A": Costo(0.3, 0.02), "B": Costo(0.04, 0.02)}), descuentos=(0.05,)
    )
    b = math.log(1.1)
    efectos = {"A": {"beta": b, "ic95": [b - 0.02, b + 0.02]}, "B": {"beta": 0.0, "ic95": [-0.01, 0.01]}}
    sin_descuento = {"beta": -0.001, "ic95": [-0.01, 0.008]}
    r = negocio.traducir(_datos(), efectos, sin_descuento, s)
    a = r["por_categoria"]["A"]
    assert a["dias_promocion"] == 2 and a["precio_lista"] == pytest.approx(10)
    # 300 · (1 − 1 / 1.1) unidades incrementales, 13.6 por día-promoción
    assert a["unidades_incrementales"] == pytest.approx(300 * (1 - 1 / 1.1))
    assert a["incrementales_por_dia_promocion"] == pytest.approx(150 * (1 - 1 / 1.1))
    bajo, alto = a["unidades_incrementales_ic95"]
    assert bajo == pytest.approx(300 * (1 - math.exp(-(b - 0.02))))
    assert bajo < a["unidades_incrementales"] < alto
    assert a["valor_maximo_sin_descuento_por_dia"] == pytest.approx(0.3 * 10 * 150 * (1 - 1 / 1.1))
    fila = a["por_descuento"][0]
    # Δπ = p · ȳ · [(m − d) − m · e^(−β)] = 10 · 150 · (0.25 − 0.3 / 1.1)
    assert fila["utilidad_por_dia_promocion"] == pytest.approx(10 * 150 * (0.25 - 0.3 / 1.1))
    assert fila["utilidad_total"] == pytest.approx(2 * fila["utilidad_por_dia_promocion"])
    assert fila["lift_equilibrio_pct"] == pytest.approx(100 * 0.05 / 0.25)
    assert fila["veredicto"] == NO_SE_PAGA  # el IC del lift (8 a 12 %) queda abajo del 20 %
    # En B el descuento supera al margen: nunca se paga
    assert r["por_categoria"]["B"]["por_descuento"][0]["lift_equilibrio_pct"] is None
    assert r["por_descuento"] == [
        {"descuento": 0.05, "origen": "rejilla", SE_PAGA: [], NO_SE_PAGA: ["A", "B"], "indeterminado": []}
    ]
    # Si el precio por unidad no baja de forma significativa, no se agrega el descuento implícito
    assert not r["descuento_implicito"]["aplica"] and r["descuento_implicito"]["descuento"] is None


def test_el_descuento_implicito_sale_del_precio_por_unidad():
    s = SupuestosNegocio(costos=MappingProxyType({"A": Costo(0.3, 0.02), "B": Costo(0.3, 0.02)}))
    efectos = {c: {"beta": 0.01, "ic95": [0.005, 0.015]} for c in ("A", "B")}
    r = negocio.traducir(_datos(), efectos, {"beta": -0.015, "ic95": [-0.02, -0.01]}, s)
    d = 1 - math.exp(-0.015)
    assert r["descuento_implicito"] == {
        "aplica": True,
        "descuento": pytest.approx(d),
        "regla": r["descuento_implicito"]["regla"],
    }
    implicito = r["por_descuento"][-1]
    assert implicito["origen"] == "implicito" and implicito["descuento"] == pytest.approx(d)
    assert [f["descuento"] for f in r["por_descuento"][:-1]] == list(negocio.DESCUENTOS)
    # Lift de equilibrio d / (m − d) ≈ 5 %; el IC del lift (0.5 a 1.5 %) queda abajo
    assert implicito[NO_SE_PAGA] == ["A", "B"]


def test_los_margenes_son_los_del_newsvendor():
    s = SupuestosNegocio()
    assert {c: s.margen(c) for c in COSTOS} == {c: costo.margen for c, costo in COSTOS.items()}
    assert s.a_json()["descuentos"] == [0.05, 0.10, 0.15, 0.20]


def test_el_veredicto_usa_el_extremo_inferior_del_ic():
    # Equilibrio: 0.05 / (0.3 − 0.05) = 20 %. El punto (25 %) queda arriba, pero el IC (15 a 35 %) lo cruza
    s = SupuestosNegocio(
        costos=MappingProxyType({"A": Costo(0.3, 0.02), "B": Costo(0.3, 0.02)}), descuentos=(0.05,)
    )
    efectos = {c: {"beta": math.log(1.25), "ic95": [math.log(1.15), math.log(1.35)]} for c in ("A", "B")}
    r = negocio.traducir(_datos(), efectos, {"beta": 0.0, "ic95": [-0.01, 0.01]}, s)
    fila = r["por_categoria"]["A"]["por_descuento"][0]
    assert fila["lift_equilibrio_pct"] == pytest.approx(20)
    assert fila["veredicto"] == negocio.INDETERMINADO
