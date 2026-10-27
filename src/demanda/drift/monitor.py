"""Los chequeos del monitor en una semana, su clasificación, las recomendaciones y la vigilancia del retador.

`revisar` es una función de (panel, pronósticos, origen): la inyección de drift solo cambia el panel.
- Semana: los días T + 1 a T + horizonte.
- Referencia: los 56 días que terminan en T, sin los días especiales (Buen Fin, temporada navideña, Semana
  Santa y festivos). Se adapta a la tendencia y a los cambios lentos, y detecta los abruptos; un drift
  gradual no lo ve, y para eso está el desempeño.
- Segmentos: el global y cada nivel de formato, región, NSE y categoría (19 con los datos actuales).
- Cada chequeo dice cuándo se puede correr: al pronosticar (en T, con lo que se sabe del futuro) o al cerrar
  (en T + horizonte, con los reales).
- Las alertas de nivel, ticket y desempeño son esperadas por calendario si la semana tiene un día especial o
  un evento del backtest (Buen Fin, Navidad o la semana posterior al Buen Fin). Las de calidad y promoción
  siempre son inesperadas: el calendario no produce nulos ni cambia el plan.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, NamedTuple, cast

import numpy as np
import polars as pl
from scipy import stats

from demanda.backtest import segmentos
from demanda.backtest.metricas import sesgo as sesgo_expr
from demanda.backtest.metricas import wape as wape_expr
from demanda.backtest.protocolo import ultima_fecha
from demanda.datos.contratos import SERIE
from demanda.drift import estadisticos
from demanda.seleccion.diebold_mariano import (
    MIN_CORTES,
    diebold_mariano,
    errores_semanales,
    perdida_por_corte,
)

# chequeo: (capa, cuándo se puede correr, medida)
CHEQUEOS = {
    "tienda_sin_datos": ("calidad", "al_cerrar", "pct_sin_units_sold"),
    "tasas_globales": ("calidad", "al_cerrar", "pct"),
    "plan_promocion": ("entradas", "al_pronosticar", "pct_con_promocion"),
    "nivel_demanda": ("entradas", "al_cerrar", "psi"),
    "ticket": ("entradas", "al_cerrar", "psi"),
    "calendario": ("entradas", "al_pronosticar", "dias_especiales"),
    "wape_global": ("desempeno", "al_cerrar", "wape_pct"),
    "sesgo_localizado": ("desempeno", "al_cerrar", "sesgo_localizado_pp"),
}
POR_CALENDARIO = ("nivel_demanda", "ticket", "wape_global", "sesgo_localizado")
EVENTOS_ESPERADOS = ("buen_fin", "navidad", "post_buen_fin")
REENTRENAN = ("nivel_demanda", "wape_global", "sesgo_localizado")
RECOMENDACIONES = {
    "revisar_captura": (
        "revisar la captura de las tiendas o el chequeo; esos días no se usan para evaluar ni reentrenar"
    ),
    "plan_de_evento": "el modelo no ha visto el evento: el pedido se revisa con el área comercial",
    "reestimar_promocion": (
        "la tasa de promoción del segmento cambió: re-estimar el efecto (E6) antes de usar el pronóstico"
    ),
    "reentrenar": "reentrenar y recalibrar los cuantiles de servicio del segmento",
    "reevaluar_campeon": "correr la regla de campeón con los cortes nuevos",
    "informativa": "drift esperado por el calendario; no requiere acción sobre el modelo",
    "sin_accion": "—",
}
CALENDARIO = ("is_buen_fin", "is_navidad_season", "is_semana_santa", "is_holiday", "holiday_name")
ATRIBUTOS = tuple(c for c in segmentos.DIMENSIONES.values() if c not in SERIE)


@dataclass(frozen=True)
class Umbrales:
    """La referencia, los umbrales de cada chequeo y los de la vigilancia del retador, fijados de antemano."""

    dias_referencia: int = 56
    horizonte: int = 7
    alfa_calidad: float = 0.01  # entre las tiendas o entre las 3 tasas (Bonferroni)
    alfa_promocion: float = 0.01  # entre los segmentos (Bonferroni)
    psi_alerta: float = 0.25
    psi_vigilar: float = 0.10
    psi_intervalos: int = estadisticos.INTERVALOS
    psi_piso: float = estadisticos.PISO
    banda_wape: float = 1.5  # × la mediana del WAPE semanal en las semanas de calibración con pronóstico
    sesgo_localizado_pp: float = 10.0
    alfa_retador: float = 0.05  # entre las revisiones semanales del backtest (Bonferroni)

    def a_json(self) -> dict[str, Any]:
        return asdict(self)


class Revision(NamedTuple):
    """Lo que el monitor encontró en una semana."""

    origen: date
    evento: str
    especiales: list[dict[str, Any]]  # los días especiales del horizonte y sus tipos
    precedentes: dict[str, bool]  # por tipo de día especial, si aparece en la historia
    medidas: list[dict[str, Any]]  # todas las medidas, con su severidad si alertan
    desempeno: dict[str, dict[str, Any]] | None  # por modelo; None sin pronósticos

    @property
    def alertas(self) -> list[dict[str, Any]]:
        """Las medidas que alertan o piden vigilar, con su capa, su clasificación y cuándo se corren."""
        especial = bool(self.especiales)
        return [
            {
                "capa": CHEQUEOS[m["chequeo"]][0],
                **m,
                "clasificacion": clasificar(m["chequeo"], especial, self.evento),
                "cuando": CHEQUEOS[m["chequeo"]][1],
            }
            for m in self.medidas
            if m["severidad"] is not None
        ]

    @property
    def sin_precedente(self) -> list[str]:
        return [t for t, visto in self.precedentes.items() if not visto]


def clasificar(chequeo: str, especial: bool, evento: str) -> str:
    """Esperada por calendario o inesperada."""
    if chequeo in POR_CALENDARIO and (especial or evento in EVENTOS_ESPERADOS):
        return "esperada"
    return "inesperada"


def es_especial() -> pl.Expr:
    return (
        pl.col("is_buen_fin") | pl.col("is_navidad_season") | pl.col("is_semana_santa") | pl.col("is_holiday")
    )


def calendario(panel: pl.DataFrame) -> pl.DataFrame:
    """Una fila por fecha con las banderas de días especiales."""
    return panel.select("date", *CALENDARIO).unique("date").sort("date")


def tipos(dia: Mapping[str, Any]) -> list[str]:
    """Los tipos de un día especial: Buen Fin, Navidad (temporada), Semana Santa o el festivo por nombre."""
    return [
        *(
            nombre
            for bandera, nombre in (
                ("is_buen_fin", "Buen Fin"),
                ("is_navidad_season", "Navidad"),
                ("is_semana_santa", "Semana Santa"),
            )
            if dia[bandera]
        ),
        *([f"festivo: {dia['holiday_name']}"] if dia["is_holiday"] else []),
    ]


def dias_especiales(
    cal: pl.DataFrame, origen: date, horizonte: int
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Los días especiales del horizonte con sus tipos y, por tipo, si aparece en la historia (fechas ≤ T)."""
    vistos = {
        t
        for dia in cal.filter((pl.col("date") <= origen) & es_especial()).iter_rows(named=True)
        for t in tipos(dia)
    }
    semana = cal.filter(
        pl.col("date").is_between(origen + timedelta(days=1), origen + timedelta(days=horizonte))
    )
    especiales = [
        {"fecha": str(dia["date"]), "tipos": tipos(dia)}
        for dia in semana.filter(es_especial()).iter_rows(named=True)
    ]
    precedentes = {t: t in vistos for dia in especiales for t in dia["tipos"]}
    return especiales, precedentes


def evento(cal: pl.DataFrame, origen: date, horizonte: int) -> str:
    """El evento semanal del backtest (`segmentos.evento`): Buen Fin o Navidad si algún día del horizonte lo
    es; si no, la semana posterior al Buen Fin si los días que terminan en T lo incluyen; si no, `resto`."""
    semana = cal.filter(
        pl.col("date").is_between(origen + timedelta(days=1), origen + timedelta(days=horizonte))
    )
    previos = cal.filter(
        pl.col("date").is_between(origen - timedelta(days=segmentos.VENTANA_POSTERIOR - 1), origen)
    )
    banderas = pl.DataFrame(
        {
            "bf": [semana["is_buen_fin"].any()],
            "nav": [semana["is_navidad_season"].any()],
            "post": [previos["is_buen_fin"].any()],
        }
    )
    return banderas.select(segmentos.evento(pl.col("bf"), pl.col("nav"), pl.col("post"))).item()


def _niveles(panel: pl.DataFrame) -> list[tuple[str, str | None, str]]:
    """Los segmentos como (segmento, columna, nivel): el global y cada nivel de las dimensiones de E3."""
    niveles: list[tuple[str, str | None, str]] = [("global", None, "global")]
    for segmento, columna in segmentos.DIMENSIONES.items():
        niveles += [(segmento, columna, n) for n in sorted(panel[columna].unique().to_list())]
    return niveles


def primera_fecha(panel: pl.DataFrame) -> date:
    return cast(date, panel["date"].min())


def _filtro(columna: str | None, nivel: str) -> pl.Expr:
    return pl.lit(True) if columna is None else pl.col(columna) == nivel


def _medida(chequeo: str, segmento: str, nivel: str, valor: float | None, **otros: Any) -> dict[str, Any]:
    return {
        "chequeo": chequeo,
        "segmento": segmento,
        "nivel": nivel,
        "modelo": otros.get("modelo"),
        "medida": otros.get("medida", CHEQUEOS[chequeo][2]),
        "valor": valor,
        "referencia": otros.get("referencia"),
        "umbral": otros.get("umbral"),
        "p_valor": otros.get("p_valor"),
        "severidad": otros.get("severidad"),
    }


def _binomial(
    chequeo: str,
    segmento: str,
    nivel: str,
    casos: int,
    n: int,
    p0: float,
    colas: int,
    umbral: float,
    **otros: Any,
) -> dict[str, Any]:
    p = estadisticos.binomial(casos, n, p0, colas)
    return _medida(
        chequeo,
        segmento,
        nivel,
        100 * casos / n if n else None,
        referencia=100 * p0,
        umbral=umbral,
        p_valor=p,
        severidad="alerta" if p < umbral else None,
        **otros,
    )


def _calidad(ref: pl.DataFrame, semana: pl.DataFrame, u: Umbrales) -> list[dict[str, Any]]:
    """Tienda sin datos (una cola, contra la tasa global de la referencia) y las tres tasas globales (dos
    colas)."""
    sin_unidades = pl.col("units_sold").is_null()
    tasa_ref = ref.select(sin_unidades.mean()).item()
    por_tienda = (
        semana.group_by("store_id").agg(sin_unidades.sum().alias("k"), pl.len().alias("n")).sort("store_id")
    )
    umbral = u.alfa_calidad / por_tienda.height
    medidas = [
        _binomial("tienda_sin_datos", "tienda", t, k, n, tasa_ref, 1, umbral)
        for t, k, n in por_tienda.iter_rows()
    ]
    existe = ~pl.col("flag_fila_faltante")
    con_efectivo = pl.col("cash_transactions").is_not_null()
    inconsistente = pl.col("cash_transactions") + pl.col("card_transactions") < pl.col("total_transactions")
    tasas = {
        "pct_sin_units_sold": (pl.lit(True), sin_unidades),
        "pct_cash_nulo": (existe, pl.col("cash_transactions").is_null()),
        "pct_conteo_inconsistente": (con_efectivo, inconsistente),
    }
    for medida, (filas, indicador) in tasas.items():
        p0 = ref.filter(filas).select(indicador.mean()).item()
        en_semana = semana.filter(filas)
        casos = int(en_semana.select(indicador.sum()).item())
        medidas.append(
            _binomial(
                "tasas_globales",
                "global",
                "global",
                casos,
                en_semana.height,
                p0,
                2,
                u.alfa_calidad / len(tasas),
                medida=medida,
            )
        )
    return medidas


def _promocion(
    ref: pl.DataFrame, semana: pl.DataFrame, niveles: Sequence[tuple[str, str | None, str]], u: Umbrales
) -> list[dict[str, Any]]:
    """% de filas con promoción en el horizonte por segmento contra su referencia, binomial de dos colas."""
    con_plan = pl.col("has_promotion").is_not_null()
    ref, semana = ref.filter(con_plan), semana.filter(con_plan)
    umbral = u.alfa_promocion / len(niveles)
    medidas = []
    for segmento, columna, nivel in niveles:
        r, s = ref.filter(_filtro(columna, nivel)), semana.filter(_filtro(columna, nivel))
        p0 = r.select(pl.col("has_promotion").mean()).item() if r.height else 0.0
        medidas.append(
            _binomial(
                "plan_promocion", segmento, nivel, int(s["has_promotion"].sum()), s.height, p0, 2, umbral
            )
        )
    return medidas


def _psi(
    ref: pl.DataFrame, semana: pl.DataFrame, niveles: Sequence[tuple[str, str | None, str]], u: Umbrales
) -> list[dict[str, Any]]:
    """PSI del nivel (units_sold) y del ticket real (amount_total / total_transactions), cada uno dividido
    entre la mediana de su serie en la referencia, por segmento. Una serie sin mediana positiva no entra, y un
    ticket sin transacciones (∞ o NaN) queda en nulo antes de la mediana."""
    medidas_por = {
        "nivel_demanda": pl.col("units_sold").cast(pl.Float64),
        "ticket": pl.when(pl.col("total_transactions") > 0).then(
            pl.col("amount_total") / pl.col("total_transactions")
        ),
    }
    medianas = ref.group_by(SERIE).agg(x.median().alias(f"mediana_{c}") for c, x in medidas_por.items())

    def razones(df: pl.DataFrame) -> pl.DataFrame:
        positiva = {c: pl.col(f"mediana_{c}") > 0 for c in medidas_por}
        return df.join(medianas, on=SERIE, how="left").with_columns(
            pl.when(positiva[c]).then(x / pl.col(f"mediana_{c}")).alias(c) for c, x in medidas_por.items()
        )

    r, s = razones(ref), razones(semana)
    medidas = []
    for segmento, columna, nivel in niveles:
        r_seg, s_seg = r.filter(_filtro(columna, nivel)), s.filter(_filtro(columna, nivel))
        for chequeo in medidas_por:
            valor = estadisticos.psi(
                r_seg[chequeo].drop_nulls().to_numpy(),
                s_seg[chequeo].drop_nulls().to_numpy(),
                u.psi_intervalos,
                u.psi_piso,
            )
            severidad = None
            if valor is not None and valor >= u.psi_alerta:
                severidad = "alerta"
            elif valor is not None and valor >= u.psi_vigilar:
                severidad = "vigilar"
            umbral = u.psi_vigilar if severidad == "vigilar" else u.psi_alerta
            medidas.append(_medida(chequeo, segmento, nivel, valor, umbral=umbral, severidad=severidad))
    return medidas


def _sesgo() -> pl.Expr:
    return 100 * sesgo_expr()


def _desempeno(
    semana: pl.DataFrame,
    pronosticos: pl.DataFrame,
    niveles: Sequence[tuple[str, str | None, str]],
    u: Umbrales,
    banda: Mapping[str, float] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]] | None]:
    """WAPE semanal global contra la banda y sesgo localizado (sesgo del segmento − sesgo global), en los
    pares completos. Los reales salen del panel; los pronósticos, de lo que se pronosticó en T."""
    if pronosticos.is_empty():
        return [], None
    reales = semana.select(*SERIE, "date", pl.col("units_sold").cast(pl.Float64).alias("y"), *ATRIBUTOS)
    pares = (
        pronosticos.select("modelo", *SERIE, "date", "yhat")
        .join(reales, on=[*SERIE, "date"], how="left", maintain_order="left")
        .group_by("modelo", *SERIE)
        .agg(
            pl.col("y").sum(),
            pl.col("yhat").sum(),
            ((pl.len() == u.horizonte) & pl.col("y").is_not_null().all()).alias("completo"),
            *(pl.col(c).first() for c in ATRIBUTOS),
        )
        .filter("completo")
        .sort("modelo", *SERIE)
    )
    medidas, desempeno = [], {}
    for modelo in pronosticos["modelo"].unique(maintain_order=True).to_list():
        del_modelo = pares.filter(pl.col("modelo") == modelo)
        if del_modelo.is_empty():  # sin pares completos no hay desempeño que medir
            desempeno[modelo] = {"wape_pct": None, "sesgo_pct": None, "pares": 0}
            continue
        wape = del_modelo.select(100 * wape_expr()).item()
        sesgo_global = del_modelo.select(_sesgo()).item()
        desempeno[modelo] = {"wape_pct": wape, "sesgo_pct": sesgo_global, "pares": del_modelo.height}
        if banda is not None and modelo in banda:
            medidas.append(
                _medida(
                    "wape_global",
                    "global",
                    "global",
                    wape,
                    modelo=modelo,
                    referencia=banda[modelo] / u.banda_wape,
                    umbral=banda[modelo],
                    severidad="alerta" if wape > banda[modelo] else None,
                )
            )
        for segmento, columna, nivel in niveles:
            if columna is None:
                continue
            del_nivel = del_modelo.filter(_filtro(columna, nivel))
            if del_nivel.is_empty():
                continue
            diferencia = del_nivel.select(_sesgo()).item() - sesgo_global
            medidas.append(
                _medida(
                    "sesgo_localizado",
                    segmento,
                    nivel,
                    diferencia,
                    modelo=modelo,
                    referencia=sesgo_global,
                    umbral=u.sesgo_localizado_pp,
                    severidad="alerta" if abs(diferencia) > u.sesgo_localizado_pp else None,
                )
            )
    return medidas, desempeno


def revisar(
    panel: pl.DataFrame,
    pronosticos: pl.DataFrame,
    origen: date,
    umbrales: Umbrales | None = None,
    banda: Mapping[str, float] | None = None,
) -> Revision:
    """Los chequeos de la semana que sigue a `origen`.

    `pronosticos` trae modelo, origen, serie, fecha y `yhat`; solo se usa lo que se pronosticó en T, y sin
    eso no hay chequeos de desempeño. `banda` es, por modelo, el WAPE semanal arriba del cual se alerta; sin
    ella ese chequeo no corre, como en las semanas de calibración que la definen.
    """
    u = umbrales or Umbrales()
    inicio, fin = origen - timedelta(days=u.dias_referencia - 1), origen + timedelta(days=u.horizonte)
    if inicio < primera_fecha(panel) or fin > ultima_fecha(panel):
        raise ValueError(
            f"la semana de {origen} no tiene {u.dias_referencia} días de referencia o su horizonte"
        )
    cal = calendario(panel)
    ventana = panel.filter(pl.col("date").is_between(inicio, fin))
    ref = ventana.filter((pl.col("date") <= origen) & ~es_especial())
    semana = ventana.filter(pl.col("date") > origen)
    niveles = _niveles(panel)

    especiales, precedentes = dias_especiales(cal, origen, u.horizonte)
    del_origen = pronosticos.filter(pl.col("origen") == origen)
    desempeno_medidas, desempeno = _desempeno(semana, del_origen, niveles, u, banda)
    medidas = [
        *_calidad(ref, semana, u),
        *_promocion(ref, semana, niveles, u),
        *_psi(ref, semana, niveles, u),
        *desempeno_medidas,
    ]
    return Revision(origen, evento(cal, origen, u.horizonte), especiales, precedentes, medidas, desempeno)


def _etiqueta(alerta: Mapping[str, Any]) -> str:
    """Dónde está la alerta: `segmento=nivel`, con la medida de las tasas y el chequeo de desempeño."""
    etiqueta = f"{alerta['segmento']}={alerta['nivel']}"
    if alerta["chequeo"] == "tasas_globales":
        etiqueta += f" · {alerta['medida']}"
    if alerta["modelo"]:
        etiqueta += f" · {alerta['chequeo']} de {alerta['modelo']}"
    elif alerta["chequeo"] in REENTRENAN:
        etiqueta += f" · {alerta['chequeo']}"
    return etiqueta


def recomendar(
    alertas: Sequence[Mapping[str, Any]], sin_precedente: Sequence[str], reevaluar: bool = False
) -> list[dict[str, Any]]:
    """Las recomendaciones de la semana, en el orden de `RECOMENDACIONES`. «Vigilar» no dispara ninguna."""
    alertas = [a for a in alertas if a["severidad"] == "alerta"]
    disparadas: dict[str, list[str]] = {}
    calidad = [_etiqueta(a) for a in alertas if a["capa"] == "calidad"]
    if calidad:
        disparadas["revisar_captura"] = calidad
    if sin_precedente:
        disparadas["plan_de_evento"] = list(sin_precedente)
    promocion = [_etiqueta(a) for a in alertas if a["chequeo"] == "plan_promocion"]
    if promocion:
        disparadas["reestimar_promocion"] = promocion
    reentrenar = [
        _etiqueta(a) for a in alertas if a["chequeo"] in REENTRENAN and a["clasificacion"] == "inesperada"
    ]
    if reentrenar:
        disparadas["reentrenar"] = reentrenar
    if reevaluar:
        disparadas["reevaluar_campeon"] = []
    if alertas and all(a["clasificacion"] == "esperada" for a in alertas):
        disparadas["informativa"] = []
    if not disparadas:
        disparadas["sin_accion"] = []
    return [
        {"codigo": c, "texto": RECOMENDACIONES[c], "por": disparadas[c]}
        for c in RECOMENDACIONES
        if c in disparadas
    ]


def cortes_necesarios(d: Sequence[float], alfa: float, maximo: int = 10_000) -> int | None:
    """El n más chico con el que la media y la desviación observadas de d_t darían p < alfa en la DM (con HLN
    y h = 1 es la t pareada, con n − 1 grados de libertad).

    Es una proyección descriptiva: supone que los cortes futuros se parecen a los vistos. None si la media es
    0, si d_t no varía o si no se llega antes de `maximo`.
    """
    x = np.asarray(d, dtype=float)
    if x.size < 2:
        return None
    media, desviacion = float(x.mean()), float(x.std(ddof=1))
    if media == 0 or desviacion == 0:
        return None
    n = np.arange(MIN_CORTES, maximo + 1)
    p = 2 * stats.t.sf(abs(media) * np.sqrt(n) / desviacion, df=n - 1)
    alcanza = np.flatnonzero(p < alfa)
    return int(n[alcanza[0]]) if alcanza.size else None


def vigilar_retador(
    pronosticos: pl.DataFrame, campeon: str, retador: str, horizonte: int, umbrales: Umbrales | None = None
) -> list[dict[str, Any]]:
    """Por cada corte cerrado del backtest, sobre los cortes cerrados hasta entonces: el WAPE semanal
    acumulado de los dos modelos, en cuántos cortes gana el retador, la DM acumulada (retador − campeón) y los
    cortes necesarios. `reevaluar` pide p < alfa / número de revisiones, con la diferencia a favor del
    retador."""
    u = umbrales or Umbrales()
    del_par = pronosticos.filter(pl.col("modelo").is_in([campeon, retador]))
    errores = errores_semanales(del_par, horizonte).sort("modelo", "origen", *SERIE)
    por_corte = perdida_por_corte(errores, retador, campeon)
    for modelo in (campeon, retador):
        absolutos = (
            errores.filter(pl.col("modelo") == modelo)
            .group_by("origen")
            .agg(pl.col("error").abs().sum().alias(f"abs_{modelo}"))
        )
        por_corte = por_corte.join(absolutos, on="origen", how="left")
    por_corte = por_corte.sort("origen").with_columns(
        (100 * pl.col(f"abs_{m}").cum_sum() / pl.col("y").cum_sum()).alias(f"wape_{m}")
        for m in (campeon, retador)
    )
    umbral = u.alfa_retador / por_corte.height
    filas = []
    for k, fila in enumerate(por_corte.iter_rows(named=True), start=1):
        d = por_corte["d"].head(k).to_list()
        dm = diebold_mariano(d) if k >= MIN_CORTES and np.std(d) > 0 else None
        filas.append(
            {
                "corte": str(fila["origen"]),
                "cortes_cerrados": k,
                "wape_acumulado_pct": {m: fila[f"wape_{m}"] for m in (campeon, retador)},
                "cortes_gana_retador": sum(x < 0 for x in d),
                "dm": None
                if dm is None
                else {
                    "media_d": dm.media,
                    "estadistico": dm.estadistico,
                    "p_valor": dm.p_valor,
                    "a_favor_de": retador if dm.media < 0 else campeon,
                },
                "cortes_necesarios": cortes_necesarios(d, u.alfa_retador),
                "umbral_p": umbral,
                "reevaluar": dm is not None and dm.p_valor < umbral and dm.media < 0,
            }
        )
    return filas
