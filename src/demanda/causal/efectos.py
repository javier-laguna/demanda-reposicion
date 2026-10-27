"""El efecto de la promoción: el modelo principal, los canales, la robustez, la heterogeneidad, la dinámica y
los efectos cruzados.

- Modelo: log(y)_{s,t} = β · promo_{s,t} + α_s + γ_t + ε, con α la serie y γ la fecha absorbidos, y errores
  agrupados por tienda con t de G − 1 gl (`absorcion`). El efecto en % es 100 · (e^β − 1), y su IC95 se
  transforma por los extremos.
- Muestra: las filas con `units_sold`, `amount_total`, `total_transactions` y `has_promotion`. Nada se imputa.
- Los rezagos y adelantos de la promoción y las promociones de las otras categorías salen del panel completo
  (la malla) antes de filtrar la muestra: un `shift` por serie es un día de calendario, y la promoción se
  observa aunque `units_sold` sea nulo.
- La heterogeneidad (6 categorías y 3 formatos) es una familia de Holm de 9 pruebas. Los canales, la dinámica
  y los efectos cruzados son secundarios: van con su IC95 y sin más corrección.
"""

import math
from typing import Any

import polars as pl

from demanda.backtest.etapa import verificar_malla
from demanda.causal.absorcion import (
    TIENDA,
    Ajuste,
    Estimacion,
    coeficiente,
    combinacion,
    estimar,
    igualdad,
)
from demanda.causal.balance import ALFA_HOLM, holm
from demanda.datos.contratos import SERIE

PROMO = "has_promotion"
Y = "log_unidades"
REZAGOS = tuple(range(1, 8))
REQUERIDAS = ("units_sold", "amount_total", "total_transactions", PROMO)
# resultado: (columna, qué mide); el primero es el principal
RESULTADOS = {
    "unidades": (Y, "log(units_sold): unidades; el único con conclusión formal"),
    "transacciones": ("log_transacciones", "log(total_transactions): tráfico"),
    "monto": ("log_monto", "log(amount_total): ingreso"),
    "precio_unitario": ("log_precio", "log(amount_total / units_sold): precio por unidad"),
    "canasta": ("log_canasta", "log(units_sold / total_transactions): unidades por transacción"),
}
DIAS_ESPECIALES = ("is_holiday", "is_buen_fin", "is_navidad_season", "is_semana_santa")
HETEROGENEIDAD = {"categoria": "category", "formato": "store_format"}
ARRASTRE, CANIBALIZACION, SIN_EFECTO = "arrastre", "canibalizacion", "sin_efecto_detectable"


def rezago(k: int) -> str:
    return f"promo_menos_{k}"


def adelanto(k: int) -> str:
    return f"promo_mas_{k}"


def preparar(panel: pl.DataFrame) -> pl.DataFrame:
    """El panel con lo que usan las estimaciones y el balance: los logaritmos de los resultados, la promoción
    de t − k y de t + k (k = 1..7) en la serie, cuántas de las otras categorías de la tienda estaban en
    promoción ese día, si las seis tienen la promoción observada y si la fila entra a la muestra."""
    verificar_malla(panel)
    promo = pl.col(PROMO)
    categorias = panel["category"].n_unique()
    return (
        panel.sort([*SERIE, "date"])
        .with_columns(
            *(promo.shift(k).over(SERIE).alias(rezago(k)) for k in REZAGOS),
            *(promo.shift(-k).over(SERIE).alias(adelanto(k)) for k in REZAGOS),
            (promo.sum().over("store_id", "date") - promo).alias("otras_en_promocion"),
            (promo.count().over("store_id", "date") == categorias).alias("tienda_dia_completo"),
            pl.all_horizontal(pl.col(c).is_not_null() for c in REQUERIDAS).alias("en_muestra"),
        )
        .with_columns(
            pl.col("units_sold").log().alias(Y),
            pl.col("total_transactions").log().alias("log_transacciones"),
            pl.col("amount_total").log().alias("log_monto"),
            (pl.col("amount_total") / pl.col("units_sold")).log().alias("log_precio"),
            (pl.col("units_sold") / pl.col("total_transactions")).log().alias("log_canasta"),
        )
    )


def muestra(preparado: pl.DataFrame) -> pl.DataFrame:
    return preparado.filter(pl.col("en_muestra"))


def pct(beta: float) -> float:
    return 100 * math.expm1(beta)


def a_json(e: Estimacion) -> dict[str, Any]:
    """Un coeficiente en puntos logarítmicos y en %, con su IC95 transformado por los extremos."""
    return {
        "beta": e.valor,
        "ee": e.ee,
        "t": e.t,
        "gl": e.gl,
        "p_valor": e.p_valor,
        "ic95": [e.ic_inf, e.ic_sup],
        "efecto_pct": pct(e.valor),
        "ic95_pct": [pct(e.ic_inf), pct(e.ic_sup)],
    }


def _ajuste_json(aj: Ajuste) -> dict[str, Any]:
    return {"n": aj.n, "clusteres": aj.gl + 1, "iteraciones_absorcion": aj.iteraciones}


def principal(m: pl.DataFrame, y: str = Y) -> tuple[dict[str, Any], Ajuste]:
    """El modelo principal sobre `y`, en JSON y el ajuste."""
    aj = estimar(m, y, [PROMO])
    return {**_ajuste_json(aj), **a_json(coeficiente(aj, PROMO))}, aj


def canales(m: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """El modelo principal con cada resultado; solo el de unidades lleva conclusión formal."""
    salida = {}
    for nombre, (columna, descripcion) in RESULTADOS.items():
        efecto, _ = principal(m, columna)
        salida[nombre] = {"resultado": descripcion, "secundario": columna != Y, **efecto}
    return salida


def robustez(m: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Las variantes del efecto principal. La inferencia por aleatorización está en `potencia`."""
    normales = m.filter(~pl.any_horizontal(pl.col(c) for c in DIAS_ESPECIALES))
    ponderada = m.with_columns(pl.col("units_sold").median().over(SERIE).cast(pl.Float64).alias("peso"))
    variantes: dict[str, tuple[str, pl.DataFrame, dict[str, Any]]] = {
        "sin_efectos_fijos": (
            "diferencia de medias de log(units_sold), sin efectos fijos",
            m,
            {"efectos": ((),)},
        ),
        "fecha_por_categoria": (
            "efectos fijos de serie y de fecha × categoría: un calendario propio por categoría",
            m,
            {"efectos": (SERIE, ("date", "category"))},
        ),
        "solo_dias_normales": (
            "sin festivos, Buen Fin, temporada navideña ni Semana Santa",
            normales,
            {},
        ),
        "ponderado_por_volumen": (
            "ponderado por la mediana de units_sold de la serie: el efecto sobre el volumen",
            ponderada,
            {"pesos": "peso"},
        ),
        "agrupado_tienda_y_fecha": (
            "errores agrupados por tienda y por fecha (Cameron, Gelbach y Miller); gl = min(G) − 1",
            m,
            {"cluster": (TIENDA, ("date",))},
        ),
    }
    salida = {}
    for nombre, (descripcion, df, opciones) in variantes.items():
        aj = estimar(df, Y, [PROMO], **opciones)
        salida[nombre] = {"descripcion": descripcion, **_ajuste_json(aj), **a_json(coeficiente(aj, PROMO))}
    return salida


def heterogeneidad(m: pl.DataFrame) -> dict[str, Any]:
    """Por categoría y por formato: el modelo principal con promo × grupo y los mismos efectos fijos, la Wald
    de igualdad dentro de cada dimensión y Holm sobre las 9 pruebas de β_g = 0."""
    salida: dict[str, Any] = {}
    for dimension, columna in HETEROGENEIDAD.items():
        niveles = sorted(m[columna].unique().to_list())
        nombres = {nivel: f"promo_{nivel}" for nivel in niveles}
        df = m.with_columns(
            (pl.col(PROMO) * (pl.col(columna) == nivel).cast(pl.Int64)).alias(nombre)
            for nivel, nombre in nombres.items()
        )
        aj = estimar(df, Y, list(nombres.values()))
        dias = dict(m.filter(pl.col(PROMO) == 1).group_by(columna).len().iter_rows())
        w = igualdad(aj, list(nombres.values()))
        salida[dimension] = {
            **_ajuste_json(aj),
            "grupos": {
                nivel: {"dias_promocion": dias.get(nivel, 0), **a_json(coeficiente(aj, nombre))}
                for nivel, nombre in nombres.items()
            },
            "igualdad": {
                "f": w.f,
                "gl": [w.gl_num, w.gl_den],
                "p_valor": w.p_valor,
                "p_sin_correccion": w.p_sin_correccion,
                "hay_heterogeneidad": w.p_valor < ALFA_HOLM,
            },
        }
    familia = [(d, g) for d in salida for g in salida[d]["grupos"]]
    ajustados = holm([salida[d]["grupos"][g]["p_valor"] for d, g in familia])
    for (d, g), p_holm in zip(familia, ajustados, strict=True):
        salida[d]["grupos"][g] |= {"p_holm": p_holm, "significativo_holm": p_holm < ALFA_HOLM}
    salida["familia_holm"] = {
        "pruebas": len(familia),
        "alfa": ALFA_HOLM,
        "significativos": [f"{d}:{g}" for d, g in familia if salida[d]["grupos"][g]["significativo_holm"]],
    }
    return salida


def dinamica(m: pl.DataFrame) -> dict[str, Any]:
    """El modelo principal con la promoción de t − 1 a t − 7, en las filas con los siete rezagos observados.
    Hay adelanto de compras si el IC95 de Σβ_k queda abajo de cero; el efecto neto es β₀ + Σβ_k."""
    rezagos = [rezago(k) for k in REZAGOS]
    df = m.filter(pl.all_horizontal(pl.col(r).is_not_null() for r in rezagos))
    aj = estimar(df, Y, [PROMO, *rezagos])
    suma = combinacion(aj, dict.fromkeys(rezagos, 1.0))
    return {
        **_ajuste_json(aj),
        "coeficientes": {
            "t": a_json(coeficiente(aj, PROMO)),
            **{f"t-{k}": a_json(coeficiente(aj, rezago(k))) for k in REZAGOS},
        },
        "suma_rezagos": a_json(suma),
        "neto": a_json(combinacion(aj, dict.fromkeys([PROMO, *rezagos], 1.0))),
        "adelanto": hay_adelanto(suma),
    }


def hay_adelanto(suma_rezagos: Estimacion) -> bool:
    """La regla de la spec: hay adelanto de compras si el IC95 de Σβ_k queda abajo de cero."""
    return suma_rezagos.ic_sup < 0


def interpretar_cruzado(e: Estimacion) -> str:
    if e.ic_inf > 0:
        return ARRASTRE
    return CANIBALIZACION if e.ic_sup < 0 else SIN_EFECTO


def cruzado(m: pl.DataFrame) -> dict[str, Any]:
    """El modelo principal más cuántas de las otras categorías de la tienda estaban en promoción ese día, en
    las tienda-días con la promoción de las seis categorías observada."""
    df = m.filter(pl.col("tienda_dia_completo"))
    aj = estimar(df, Y, [PROMO, "otras_en_promocion"])
    otras = coeficiente(aj, "otras_en_promocion")
    return {
        **_ajuste_json(aj),
        "propia": a_json(coeficiente(aj, PROMO)),
        "otras_categorias": a_json(otras),
        "otras_en_promocion_media": df["otras_en_promocion"].cast(pl.Float64).sum() / df.height,
        "interpretacion": interpretar_cruzado(otras),
    }


def transacciones_registradas(m: pl.DataFrame) -> dict[str, Any]:
    """Diagnóstico, no pre-registrado: el canal de tráfico con los conteos por medio de pago,
    log(cash_transactions + card_transactions), en las filas con efectivo observado, y la razón media
    total_transactions / (cash + card) con y sin promoción. Si la razón cambia con la promoción, el canal de
    `total_transactions` mezcla el efecto con el registro de los conteos."""
    df = m.filter(pl.col("cash_transactions").is_not_null()).with_columns(
        (pl.col("cash_transactions") + pl.col("card_transactions")).log().alias("log_efectivo_tarjeta")
    )
    efecto, _ = principal(df, "log_efectivo_tarjeta")
    razon = (
        pl.col("total_transactions") / (pl.col("cash_transactions") + pl.col("card_transactions"))
    ).mean()
    razones = dict(df.group_by(PROMO).agg(razon).iter_rows())
    return {
        "resultado": "log(cash_transactions + card_transactions): tráfico con los conteos por medio de pago",
        "secundario": True,
        "pre_registrado": False,
        "razon_total_sobre_efectivo_tarjeta": {
            "con_promocion": razones.get(1),
            "sin_promocion": razones.get(0),
        },
        **efecto,
    }
