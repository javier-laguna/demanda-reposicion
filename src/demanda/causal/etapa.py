"""La etapa del experimento de promociones: del panel a `promociones.json`.

En el orden de la spec: primero el balance (B1 a B5, con Holm); después el efecto principal, los canales, la
robustez con la inferencia por aleatorización y la calibración A/A; la heterogeneidad, la dinámica y los
efectos cruzados; la potencia y la traducción a negocio. `conclusion` trae las reglas ya evaluadas.

No lleva filas ni series, solo agregados y estadísticos. Dos corridas lo escriben igual: las permutaciones
usan semillas fijas y los números van con 6 cifras significativas (2 decimales desde un millón).
"""

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from demanda.causal import balance, efectos, negocio, potencia
from demanda.causal.efectos import PROMO, REZAGOS, Y, adelanto, rezago
from demanda.causal.negocio import SupuestosNegocio
from demanda.causal.potencia import Parametros
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256

CIFRAS = 6
DEFINICIONES = {
    "muestra": (
        "filas con units_sold, amount_total, total_transactions y has_promotion no nulos; serie = tienda × "
        "categoría; clúster = tienda"
    ),
    "beta": "coeficiente de has_promotion en puntos logarítmicos",
    "efecto_pct": "100 · (e^β − 1); su IC95 transforma los extremos del IC95 de β",
    "ee": (
        "error estándar agrupado por tienda (CRV1), con G/(G − 1) · (N − 1)/(N − K) y K sin los efectos "
        "fijos absorbidos"
    ),
    "gl": "grados de libertad de la t y del denominador de la F: clústeres − 1",
    "absorcion": "proyecciones alternadas por serie y por fecha hasta que el cambio máximo baje de 1e-10",
    "p_holm": "valor p ajustado por Holm dentro de su familia",
    "familia_balance": "las 13 χ² de B2, las 2 de B4 y las 2 de B5",
    "familia_heterogeneidad": "las 9 pruebas de β_g = 0: 6 categorías y 3 formatos",
    "igualdad": "F de Wald de coeficientes iguales dentro de la dimensión; hay heterogeneidad si p < 0.05",
    "wald": (
        "F = W / q con la covarianza agrupada; p con la corrección de Hotelling, F · (G − q)/(G − 1) contra "
        "F(q, G − q); p_sin_correccion contra F(q, G − 1)"
    ),
    "p_aleatorizacion": (
        "(1 + #{|β_b| ≥ |β_obs|}) / (B + 1), con B permutaciones de la promoción entre las filas de cada "
        "fecha"
    ),
    "calibracion_aa": (
        "tasa de |t| > t₀.₉₇₅,gl en permutaciones sin efecto; si sale del intervalo binomial del 99 % "
        "alrededor de 5 %, la inferencia principal es la de aleatorización"
    ),
    "mde": "(t₀.₉₇₅,gl + t₀.₈,gl) · EE, en log; mde_pct = 100 · (e^mde − 1)",
    "suma_rezagos": "Σβ_k de t − 1 a t − 7; hay adelanto de compras si su IC95 queda abajo de cero",
    "neto": "β₀ + Σβ_k: el efecto de un día de promoción sobre las ventas de esa semana",
    "otras_categorias": (
        "coeficiente del número de las otras cinco categorías de la tienda en promoción el mismo día: "
        "arrastre si su IC95 queda arriba de cero, canibalización si queda abajo"
    ),
    "dia_promocion": "una fila tienda × categoría × día con has_promotion = 1",
    "unidades_incrementales": "Σ y · (1 − e^(−β_c)) en los días con promoción de la categoría",
    "precio_lista": "Σ amount_total / Σ units_sold en los días sin promoción de la categoría",
    "utilidad_por_dia_promocion": (
        "p · y₀ · [(m − d) · L − m], en pesos, con y₀ = ȳ · e^(−β), L = e^β y ȳ las unidades medias por "
        "día-promoción"
    ),
    "lift_equilibrio_pct": "100 · d / (m − d); nulo si d ≥ m, porque la promoción nunca se paga",
    "veredicto": (
        "se_paga si el extremo inferior del IC95 del lift supera el de equilibrio; no_se_paga si el superior "
        "queda abajo o si d ≥ m; indeterminado en otro caso"
    ),
    "valor_maximo_sin_descuento_por_dia": (
        "m · p · unidades incrementales por día-promoción, en pesos: lo más que puede costar una promoción "
        "sin descuento para pagarse"
    ),
}


def redondear(x: Any) -> Any:
    """Redondea todos los números de una estructura JSON: 6 cifras significativas, o 2 decimales desde un
    millón. Los no finitos quedan en nulo."""
    if isinstance(x, Mapping):
        return {k: redondear(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [redondear(v) for v in x]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        if not math.isfinite(v):
            return None
        return float(f"{v:.{CIFRAS}g}") if abs(v) < 1e6 else round(v, 2)
    return x


def _muestra(preparado: pl.DataFrame, m: pl.DataFrame) -> dict[str, Any]:
    existentes = ~pl.col("flag_fila_faltante")
    return {
        "filas_panel": preparado.height,
        "filas": m.height,
        "excluidas": {
            "fila_faltante": int(preparado["flag_fila_faltante"].sum()),
            "units_sold_nulo": preparado.filter(existentes & pl.col("units_sold").is_null()).height,
            "otras_medidas_nulas": preparado.filter(
                existentes & pl.col("units_sold").is_not_null() & ~pl.col("en_muestra")
            ).height,
        },
        "series": m.select(SERIE).n_unique(),
        "tiendas": m["store_id"].n_unique(),
        "fechas": m["date"].n_unique(),
        "desde": str(m["date"].min()),
        "hasta": str(m["date"].max()),
        "dias_promocion": int(m[PROMO].sum()),
        "tasa_promocion_pct": 100 * int(m[PROMO].sum()) / m.height,
        "unidad": "serie = tienda × categoría; clúster = tienda",
    }


def _advertir_conteos(canales: dict[str, Any]) -> None:
    """Agrega a tráfico y canasta la razón total / (cash + card) con y sin promoción del diagnóstico."""
    razon = canales["transacciones_efectivo_tarjeta"]["razon_total_sobre_efectivo_tarjeta"]
    con, sin = razon["con_promocion"], razon["sin_promocion"]
    if con is None or sin is None:
        return
    texto = (
        f"total_transactions es en promedio {con:.4f} × (cash + card) en los días con promoción y "
        f"{sin:.4f} × sin ella: si difieren, el canal mezcla el efecto con el registro de los conteos; ver "
        "transacciones_efectivo_tarjeta"
    )
    for canal in ("transacciones", "canasta"):
        canales[canal]["advertencia"] = texto


def _potencia(
    principal: Mapping[str, Any], het: Mapping[str, Any], semanal: dict[str, Any], p: Parametros
) -> dict[str, Any]:
    gl = principal["gl"]
    return {
        "diseno_actual": {
            "descripcion": "la asignación diaria por fila de los datos, con el modelo principal",
            "alfa": p.alfa,
            "potencia": p.potencia,
            "factor": potencia.factor_mde(gl, p.alfa, p.potencia),
            "global": potencia.mde(principal["ee"], gl, p.alfa, p.potencia),
            **{
                dimension: {
                    g: potencia.mde(v["ee"], v["gl"], p.alfa, p.potencia)
                    for g, v in het[dimension]["grupos"].items()
                }
                for dimension in efectos.HETEROGENEIDAD
            },
        },
        "diseno_semanal": semanal,
    }


def _conclusion(
    bal: Mapping[str, Any],
    principal: Mapping[str, Any],
    het: Mapping[str, Any],
    din: Mapping[str, Any],
    cru: Mapping[str, Any],
    neg: Mapping[str, Any],
    alfa: float,
) -> dict[str, Any]:
    """Las reglas de la spec ya evaluadas y una frase hecha con plantilla."""
    pruebas = {f["prueba"]: f for f in bal["familia"]["pruebas"]}
    if bal["se_sostiene"]:
        balance_txt = "La aleatorización se sostiene con la regla de la spec."
    else:
        balance_txt = (
            "La aleatorización no se sostiene con la regla de la spec: falla "
            f"{', '.join(bal['fallan'])}. Los efectos fijos absorben el desbalance de tienda, serie y "
            "calendario."
        )
        conteo = pruebas.get("B5:conteo_inconsistente")
        if conteo is not None and conteo["prueba"] in bal["fallan"]:
            balance_txt += (
                " flag_conteo_inconsistente marca el "
                f"{conteo['tasa_con_promocion_pct']:.4g} % de las filas con promoción y el "
                f"{conteo['tasa_sin_promocion_pct']:.4g} % de las demás: la promoción mueve la bandera."
            )
        if bal["asignacion_se_sostiene"]:
            balance_txt += (
                " Esa prueba mide el registro del mismo día y no la asignación. Las pruebas de la asignación"
                " pasan todas, así que la promoción se asignó al azar. La separación se hizo después de ver"
                " el resultado."
            )
    ic = principal["ic95_pct"]
    significativos = het["familia_holm"]["significativos"]
    efecto_txt = (
        f"La promoción cambia las unidades en {principal['efecto_pct']:+.2f} % (IC95 de {ic[0]:+.2f} a "
        f"{ic[1]:+.2f} %; p = {principal['p_valor_principal']:.3g} con la inferencia "
        f"{principal['inferencia_principal'].replace('_', ' ')}), "
        f"{'' if principal['significativo'] else 'no '}significativo al {100 * alfa:g} %."
    )
    verbo = "es distinto" if len(significativos) == 1 else "son distintos"
    het_txt = (
        f"Con Holm, {', '.join(significativos)} {verbo} de cero."
        if significativos
        else "Con Holm, ningún grupo es distinto de cero."
    )
    suma = din["suma_rezagos"]
    din_txt = (
        f"{'Hay' if din['adelanto'] else 'No hay'} adelanto de compras (Σ rezagos "
        f"{suma['efecto_pct']:+.2f} %, IC95 de {suma['ic95_pct'][0]:+.2f} a {suma['ic95_pct'][1]:+.2f} %)."
    )
    cruzado_txt = f"Efecto sobre las otras categorías: {cru['interpretacion'].replace('_', ' ')}."
    return {
        "aleatorizacion_se_sostiene": bal["se_sostiene"],
        "pruebas_que_fallan": bal["fallan"],
        "asignacion_se_sostiene": bal["asignacion_se_sostiene"],
        "inferencia_principal": principal["inferencia_principal"],
        "efecto_principal_pct": principal["efecto_pct"],
        "efecto_principal_ic95_pct": ic,
        "p_valor_principal": principal["p_valor_principal"],
        "efecto_principal_significativo": principal["significativo"],
        "heterogeneidad_significativa_holm": significativos,
        "hay_heterogeneidad": {d: het[d]["igualdad"]["hay_heterogeneidad"] for d in efectos.HETEROGENEIDAD},
        "adelanto_de_compras": din["adelanto"],
        "efecto_cruzado": cru["interpretacion"],
        "descuento_implicito": neg["descuento_implicito"]["descuento"],
        "se_paga": {f"{100 * f['descuento']:.4g}%": f[negocio.SE_PAGA] for f in neg["por_descuento"]},
        "texto": " ".join((balance_txt, efecto_txt, het_txt, din_txt, cruzado_txt)),
    }


def analizar(
    panel: pl.DataFrame,
    p: Parametros | None = None,
    s: SupuestosNegocio | None = None,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """El contenido de `promociones.json`, sin la fuente ni redondear."""
    p, s = p or Parametros(), s or SupuestosNegocio()

    def avisar(mensaje: str) -> None:
        if progreso is not None:
            progreso(mensaje)

    preparado = efectos.preparar(panel)
    m = efectos.muestra(preparado)
    avisar("balance: B1 a B5")
    bal = balance.evaluar(preparado, m, [rezago(1), rezago(7)], [adelanto(k) for k in REZAGOS], Y)

    avisar("efecto principal, canales y robustez")
    principal, _ = efectos.principal(m)
    canales = {**efectos.canales(m), "transacciones_efectivo_tarjeta": efectos.transacciones_registradas(m)}
    _advertir_conteos(canales)
    robustez = efectos.robustez(m)
    dis = potencia.diseno(m, Y, PROMO)
    avisar(f"inferencia por aleatorización: {p.permutaciones:,} permutaciones")
    robustez["aleatorizacion"] = potencia.aleatorizacion(dis, principal["beta"], p)
    avisar(f"calibración A/A: {p.placebos_aa:,} permutaciones")
    aa = potencia.calibracion_aa(dis, p)
    fisher = robustez["aleatorizacion"]["p_valor"]
    p_principal = principal["p_valor"] if aa["inferencia_principal"] == "agrupada_por_tienda" else fisher
    principal = {
        "modelo": "log(units_sold) = β · has_promotion + α_serie + γ_fecha + ε",
        "series": m.select(SERIE).n_unique(),
        **principal,
        "p_aleatorizacion": fisher,
        "inferencia_principal": aa["inferencia_principal"],
        "p_valor_principal": p_principal,
        "significativo": p_principal < p.alfa,
    }

    avisar("heterogeneidad, dinámica y efectos cruzados")
    het = efectos.heterogeneidad(m)
    din = efectos.dinamica(m)
    cru = efectos.cruzado(m)
    avisar(f"diseño semanal: {p.placebos_semanal:,} asignaciones placebo")
    pot = _potencia(principal, het, potencia.diseno_semanal(m, p), p)
    neg = negocio.traducir(m, het["categoria"]["grupos"], canales["precio_unitario"], s)

    return {
        "muestra": _muestra(preparado, m),
        "balance": bal,
        "efecto_principal": principal,
        "canales": canales,
        "robustez": robustez,
        "calibracion_aa": aa,
        "heterogeneidad": het,
        "dinamica": din,
        "cruzado": cru,
        "potencia": pot,
        "negocio": neg,
        "supuestos": {
            "parametros": p.a_json(),
            "negocio": s.a_json(),
            "estimacion": (
                "log-OLS con efectos fijos de serie y de fecha absorbidos; errores agrupados por tienda; t "
                "con G − 1 gl"
            ),
            "no_anticipacion": "B3 supone que los clientes no conocen las promociones por adelantado",
            "sin_contaminacion": "la promoción de una tienda no cambia las ventas de otra",
            "potencia_semanal": pot["diseno_semanal"]["metodo"],
        },
        "definiciones": DEFINICIONES,
        "conclusion": _conclusion(bal, principal, het, din, cru, neg, p.alfa),
    }


def ejecutar(
    panel_path: Path,
    salida_path: Path,
    p: Parametros | None = None,
    s: SupuestosNegocio | None = None,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Lee el panel y escribe `promociones.json`."""
    panel = pl.read_parquet(panel_path)
    resultado = redondear({"fuente": {"panel_sha256": sha256(panel_path)}, **analizar(panel, p, s, progreso)})
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    salida_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n")
    return resultado
