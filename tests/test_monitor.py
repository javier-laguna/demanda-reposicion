"""El monitor sobre paneles sintéticos: semana limpia, escenarios inyectados, calendario, sesgo localizado,
vigilancia del retador y la etapa de punta a punta."""

import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats

from demanda.backtest import bucle
from demanda.backtest.protocolo import Protocolo
from demanda.drift import etapa, inyeccion, monitor
from demanda.modelos import Contexto, crear
from demanda.seleccion.diebold_mariano import diebold_mariano, errores_semanales, perdida_por_corte

T = date(2023, 3, 4)  # la semana va del 03-05 al 03-11; la referencia, del 01-08 al 03-04
MODELOS = ["sistema_actual", "ma7"]


@pytest.fixture(scope="module")
def limpio(panel_monitor):
    """El panel y lo que el sistema actual y la MA7 pronosticaron en T."""
    modelos = crear(Contexto(factor_senal=0.53), MODELOS)
    return panel_monitor, bucle.correr(panel_monitor, modelos, Protocolo(), origenes=[T]).pronosticos


def _inesperadas(rev: monitor.Revision) -> list[dict]:
    return [a for a in rev.alertas if a["severidad"] == "alerta" and a["clasificacion"] == "inesperada"]


def _codigos(rev: monitor.Revision) -> list[str]:
    return [r["codigo"] for r in monitor.recomendar(rev.alertas, rev.sin_precedente)]


def test_en_un_panel_limpio_no_hay_alertas_inesperadas(limpio):
    p, pron = limpio
    rev = monitor.revisar(p, pron, T)
    assert rev.evento == "resto" and rev.especiales == []
    assert _inesperadas(rev) == []
    assert _codigos(rev) == ["sin_accion"]
    # Corren las tres capas: 20 tiendas, 3 tasas y 19 segmentos; el sesgo, en los 18 no globales por modelo.
    # Sin banda no corre el WAPE
    chequeos = Counter(m["chequeo"] for m in rev.medidas)
    assert chequeos == {
        "tienda_sin_datos": 20,
        "tasas_globales": 3,
        "plan_promocion": 19,
        "nivel_demanda": 19,
        "ticket": 19,
        "sesgo_localizado": 2 * 18,
    }
    assert set(rev.desempeno or {}) == set(MODELOS)
    # Sin pronósticos del origen no hay desempeño
    assert monitor.revisar(p, pron.filter(pl.col("origen") != T), T).desempeno is None


@pytest.mark.parametrize("nombre", list(inyeccion.ESCENARIOS))
def test_cada_escenario_dispara_su_chequeo_y_su_recomendacion(limpio, nombre):
    p, pron = limpio
    escenario = inyeccion.ESCENARIOS[nombre]
    rev = monitor.revisar(escenario.inyectar(p, T, 7), pron, T)
    for esperada in escenario.esperadas(p):
        assert all(inyeccion.encontradas(rev.alertas, esperada, MODELOS).values()), esperada
    assert escenario.recomendacion in _codigos(rev)


def test_las_inyecciones_solo_tocan_la_semana_y_su_segmento(limpio):
    p, _ = limpio
    semana = (p["date"] > T) & (p["date"] <= T + timedelta(days=7))

    falla = inyeccion.falla_pos(p, T)
    nuevos = falla["units_sold"].is_null() & p["units_sold"].is_not_null()
    tocadas = p.filter(nuevos)
    assert set(tocadas["store_id"]) == {"STR_001"}
    assert set(tocadas["date"]) <= {T + timedelta(days=d) for d in (2, 3, 4)}
    tienda = falla.filter(
        (pl.col("store_id") == "STR_001")
        & pl.col("date").is_between(T + timedelta(days=2), T + timedelta(days=4))
    )
    assert tienda["units_sold"].null_count() == 18 and tienda["flag_nulo_units_sold"].all()

    salto = inyeccion.salto_nivel(p, T, 7, 1.3, "region", "Centro")
    cambio = (salto["units_sold"] != p["units_sold"]).fill_null(False)
    assert set(p.filter(cambio)["region"]) == {"Centro"} and semana.filter(cambio).all()
    centro = p.filter(semana & (p["region"] == "Centro"))
    assert salto.filter(semana & (p["region"] == "Centro"))["units_sold"].sum() == pytest.approx(
        1.3 * centro["units_sold"].sum(), rel=1e-3
    )

    promo = inyeccion.cambio_promocion(p, T, 7)
    cambio = (promo["has_promotion"] != p["has_promotion"]).fill_null(False)
    assert set(p.filter(cambio)["category"]) == {"Bebidas"} and semana.filter(cambio).all()
    assert (p.filter(cambio)["has_promotion"] == 0).all()
    bebidas = promo.filter(semana & (p["category"] == "Bebidas"))
    assert bebidas["has_promotion"].mean() == pytest.approx(0.40, abs=1 / bebidas.height)
    # Con la misma semilla se eligen las mismas filas
    assert inyeccion.cambio_promocion(p, T, 7).equals(promo)


def _con_festivo(p: pl.DataFrame, dia: date, nombre: str) -> pl.DataFrame:
    es = pl.col("date") == dia
    return p.with_columns(
        (pl.col("is_holiday") | es).alias("is_holiday"),
        pl.when(es).then(pl.lit(nombre)).otherwise(pl.col("holiday_name")).alias("holiday_name"),
    )


def test_con_un_dia_especial_la_alerta_de_nivel_es_esperada(limpio):
    p, pron = limpio
    fiesta = date(2023, 3, 8)
    rev = monitor.revisar(
        inyeccion.salto_nivel(_con_festivo(p, fiesta, "Fiesta"), T, 7, 1.3, "region", "Centro"), pron, T
    )
    nivel = [a for a in rev.alertas if a["chequeo"] == "nivel_demanda" and a["nivel"] == "Centro"]
    assert nivel and all(a["clasificacion"] == "esperada" and a["cuando"] == "al_cerrar" for a in nivel)
    assert rev.especiales == [{"fecha": "2023-03-08", "tipos": ["festivo: Fiesta"]}]
    assert rev.precedentes == {"festivo: Fiesta": False}
    # El festivo no tiene precedente: plan de evento, y el salto es esperado, así que no se reentrena
    assert _codigos(rev) == ["plan_de_evento", "informativa"]

    # Año Nuevo sí tiene precedente (2023-01-01 está en la historia): no hay plan de evento
    rev = monitor.revisar(_con_festivo(p, fiesta, "Año Nuevo"), pron, T)
    assert rev.precedentes == {"festivo: Año Nuevo": True} and "plan_de_evento" not in _codigos(rev)


def test_la_referencia_no_usa_los_dias_especiales(limpio):
    p, pron = limpio
    # Diez días de la referencia pasan a ser festivos; en una copia, además, venden el triple
    dias = [date(2023, 2, 1) + timedelta(days=k) for k in range(10)]
    festivos = p.with_columns(
        (pl.col("is_holiday") | pl.col("date").is_in(dias)).alias("is_holiday"),
        pl.when(pl.col("date").is_in(dias))
        .then(pl.lit("Fiesta"))
        .otherwise(pl.col("holiday_name"))
        .alias("holiday_name"),
    )
    triple = festivos.with_columns(
        pl.when(pl.col("date").is_in(dias))
        .then(pl.col("units_sold") * 3)
        .otherwise(pl.col("units_sold"))
        .alias("units_sold")
    )

    def valores(panel: pl.DataFrame) -> dict:
        return {
            (m["chequeo"], m["segmento"], m["nivel"], m["modelo"], m["medida"]): m["valor"]
            for m in monitor.revisar(panel, pron, T).medidas
        }

    assert valores(triple) == valores(festivos)
    assert valores(festivos) != valores(p)


def test_la_semana_del_buen_fin_y_la_siguiente_son_eventos(panel_monitor):
    cal = monitor.calendario(
        panel_monitor.with_columns((pl.col("date") == date(2023, 3, 6)).alias("is_buen_fin"))
    )
    assert monitor.evento(cal, T, 7) == "buen_fin"
    assert monitor.evento(cal, date(2023, 3, 9), 7) == "post_buen_fin"
    assert monitor.evento(cal, date(2023, 2, 25), 7) == "resto"
    assert monitor.clasificar("nivel_demanda", False, "post_buen_fin") == "esperada"
    assert monitor.clasificar("tienda_sin_datos", True, "buen_fin") == "inesperada"
    assert monitor.clasificar("plan_promocion", True, "navidad") == "inesperada"


def test_el_sesgo_localizado_ignora_un_choque_que_mueve_igual_todos_los_segmentos(limpio):
    p, pron = limpio
    en_semana = pl.col("date") > T
    choque = p.with_columns(
        pl.when(en_semana)
        .then((pl.col("units_sold") * 1.3).round().cast(pl.Int64))
        .otherwise(pl.col("units_sold"))
        .alias("units_sold")
    )
    rev, base = monitor.revisar(choque, pron, T), monitor.revisar(p, pron, T)
    assert rev.desempeno is not None and base.desempeno is not None
    for m in MODELOS:
        assert rev.desempeno[m]["sesgo_pct"] < base.desempeno[m]["sesgo_pct"] - 15
    assert [a for a in rev.alertas if a["chequeo"] == "sesgo_localizado"] == []
    # El mismo choque solo en Centro sí se localiza
    region = monitor.revisar(inyeccion.salto_nivel(p, T, 7, 1.3, "region", "Centro"), pron, T)
    localizado = [a for a in region.alertas if a["chequeo"] == "sesgo_localizado"]
    assert {(a["nivel"], a["modelo"]) for a in localizado} >= {("Centro", m) for m in MODELOS}


def test_la_banda_de_wape_dispara_reentrenar(limpio):
    p, pron = limpio
    base = monitor.revisar(p, pron, T).desempeno
    assert base is not None
    banda = {
        "sistema_actual": base["sistema_actual"]["wape_pct"] - 0.01,
        "ma7": base["ma7"]["wape_pct"] + 0.01,
    }
    rev = monitor.revisar(p, pron, T, banda=banda)
    wape = [a for a in rev.alertas if a["chequeo"] == "wape_global"]
    assert [(a["modelo"], a["clasificacion"]) for a in wape] == [("sistema_actual", "inesperada")]
    assert "reentrenar" in _codigos(rev)


def test_vigilar_no_dispara_recomendaciones():
    vigilar = {
        "capa": "entradas",
        "chequeo": "nivel_demanda",
        "segmento": "region",
        "nivel": "Sur",
        "modelo": None,
        "medida": "psi",
        "severidad": "vigilar",
        "clasificacion": "inesperada",
    }
    assert [r["codigo"] for r in monitor.recomendar([vigilar], [])] == ["sin_accion"]
    assert [r["codigo"] for r in monitor.recomendar([vigilar], [], reevaluar=True)] == ["reevaluar_campeon"]


def _pronosticos(ventaja: float, semilla: int = 0) -> pl.DataFrame:
    """2 modelos × 17 cortes semanales × 4 series × 7 días. El error relativo del retador es `ventaja` más
    chico que el del campeón, en promedio."""
    rng = np.random.default_rng(semilla)
    filas = []
    for k in range(17):
        origen = date(2023, 1, 5) + timedelta(days=7 * k)
        for s in range(4):
            for h in range(1, 8):
                y = float(rng.integers(50, 150))
                for modelo, escala in (("campeon", 0.3), ("retador", 0.3 - ventaja)):
                    yhat = y * (1 + escala * rng.standard_normal())
                    filas.append(
                        (modelo, origen, f"STR_{s:03d}", "Ropa", origen + timedelta(days=h), h, y, yhat)
                    )
    columnas = ["modelo", "origen", "store_id", "category", "date", "h", "y", "yhat"]
    return pl.DataFrame(filas, schema=columnas, orient="row")


def test_con_los_17_cortes_la_dm_acumulada_es_la_directa():
    pron = _pronosticos(ventaja=0.05)
    filas = monitor.vigilar_retador(pron, "campeon", "retador", 7)
    assert [f["cortes_cerrados"] for f in filas] == list(range(1, 18))
    assert filas[1]["dm"] is None and filas[2]["dm"] is not None  # la DM pide 3 cortes
    d = perdida_por_corte(errores_semanales(pron, 7), "retador", "campeon")["d"].to_list()
    directa = diebold_mariano(d)
    final = filas[-1]["dm"]
    assert final["p_valor"] == pytest.approx(directa.p_valor)
    assert final["estadistico"] == pytest.approx(directa.estadistico)
    assert filas[-1]["cortes_gana_retador"] == sum(x < 0 for x in d)
    assert filas[-1]["umbral_p"] == pytest.approx(0.05 / 17)
    # El WAPE acumulado al último corte es el global de cada modelo
    errores = errores_semanales(pron, 7)
    for m in ("campeon", "retador"):
        del_modelo = errores.filter(pl.col("modelo") == m)
        assert filas[-1]["wape_acumulado_pct"][m] == pytest.approx(
            del_modelo.select(100 * pl.col("error").abs().sum() / pl.col("y").sum()).item()
        )


def test_reevaluar_pide_bonferroni_y_la_diferencia_a_favor_del_retador():
    a_favor = monitor.vigilar_retador(_pronosticos(ventaja=0.25), "campeon", "retador", 7)
    assert a_favor[-1]["reevaluar"] and a_favor[-1]["dm"]["p_valor"] < 0.05 / 17
    # Una semana con p < 0.05 pero arriba de 0.05 / 17 no basta
    assert all(f["reevaluar"] == (f["dm"] is not None and f["dm"]["p_valor"] < 0.05 / 17) for f in a_favor)
    # Si el que gana es el campeón, no se reevalúa por pequeña que sea la p
    en_contra = monitor.vigilar_retador(_pronosticos(ventaja=0.25), "retador", "campeon", 7)
    assert en_contra[-1]["dm"]["p_valor"] < 0.05 / 17 and not any(f["reevaluar"] for f in en_contra)


def test_cortes_necesarios_es_el_primer_n_con_p_menor_a_alfa():
    d = [-3.0, 1.0, -2.0, 0.5, -1.5, -0.5]
    n = monitor.cortes_necesarios(d, 0.05)
    media, desviacion = np.mean(d), np.std(d, ddof=1)

    def p(k: int) -> float:
        return float(2 * stats.t.sf(abs(media) * np.sqrt(k) / desviacion, k - 1))

    assert n is not None and p(n) < 0.05 <= p(n - 1)
    # Con los cortes observados, la proyección es la DM: p(6) es la de la prueba
    assert p(len(d)) == pytest.approx(diebold_mariano(d).p_valor)
    assert monitor.cortes_necesarios([1.0, 1.0, 1.0], 0.05) is None
    assert monitor.cortes_necesarios([1.0, -1.0], 0.05) is None


VALIDACION = (date(2023, 3, 18), date(2023, 3, 25))


def test_drift_de_punta_a_punta_es_reproducible(artefactos_e5, tmp_path):
    a = artefactos_e5
    rutas = (a.panel, a.pronosticos, a.metricas, a.seleccion, a.calibracion)
    r = etapa.ejecutar(*rutas, tmp_path / "a.json", semanas_validacion=VALIDACION)
    etapa.ejecutar(*rutas, tmp_path / "b.json", semanas_validacion=VALIDACION)
    texto = (tmp_path / "a.json").read_text()
    assert texto == (tmp_path / "b.json").read_text()
    assert json.loads(texto) == json.loads(json.dumps(r))

    # 4 semanas de calibración (desde el primer jueves con 56 días de historia) y los 4 cortes
    origenes = [s["origen"] for s in r["semanas"]]
    assert origenes == [str(date(2023, 2, 25) + timedelta(days=7 * k)) for k in range(8)]
    assert [s["periodo"] for s in r["semanas"]] == ["calibracion"] * 4 + ["backtest"] * 4
    assert [s["evento"] for s in r["semanas"][4:]] == ["resto", "buen_fin", "post_buen_fin", "navidad"]
    assert all(s["recomendaciones"] for s in r["semanas"])
    # La banda sale de las 2 semanas con pronóstico; el WAPE corre solo en el backtest y el sesgo también en
    # las semanas de calibración con pronóstico
    assert {b["semanas"] for b in r["configuracion"]["banda_wape"].values()} == {2}
    cal, bt = r["resumen"]["calibracion"]["por_chequeo"], r["resumen"]["backtest"]["por_chequeo"]
    assert cal["wape_global"]["semanas_evaluadas"] == 0 and bt["wape_global"]["semanas_evaluadas"] == 4
    assert (
        cal["sesgo_localizado"]["semanas_evaluadas"] == 2
        and cal["tienda_sin_datos"]["semanas_evaluadas"] == 4
    )
    capas = r["resumen"]["backtest"]["por_capa"]
    assert list(capas) == ["calidad", "entradas", "desempeno"]
    assert capas["desempeno"]["alertas"] == bt["wape_global"]["alertas"] + bt["sesgo_localizado"]["alertas"]
    # Bonferroni: lo que dice la configuración es el umbral que se aplicó en las alertas y en la vigilancia
    bonferroni = r["configuracion"]["bonferroni"]
    assert bonferroni == {
        "tienda_sin_datos": {"alfa": 0.01, "comparaciones": 10, "umbral_p": 0.001},
        "tasas_globales": {"alfa": 0.01, "comparaciones": 3, "umbral_p": 0.003333},
        "plan_promocion": {"alfa": 0.01, "comparaciones": 19, "umbral_p": 0.0005263},
        "vigilancia_retador": {"alfa": 0.05, "comparaciones": 4, "umbral_p": 0.0125},
    }
    for s in r["semanas"]:
        for al in s["alertas"]:
            if al["chequeo"] in bonferroni:
                assert al["umbral"] == bonferroni[al["chequeo"]]["umbral_p"]
    assert {f["umbral_p"] for f in r["vigilancia_retador"]["por_corte"]} == {0.0125}

    # La DM acumulada al último corte es la de seleccion.json
    v = r["vigilancia_retador"]
    assert v["coincide_con_seleccion"] is True and len(v["por_corte"]) == 4
    seleccion = json.loads(a.seleccion.read_text())
    assert [r["campeon"], r["retador"]] == [
        seleccion["campeon"],
        *(m for m in seleccion["modelos"] if m != seleccion["campeon"]),
    ]

    val = r["validacion"]
    assert [(e["escenario"], e["origen"]) for e in val["escenarios"]] == [
        (n, str(o)) for o in VALIDACION for n in inyeccion.ESCENARIOS
    ]
    resumen = val["potencia"]["resumen_salto_nivel"]
    assert [(x["segmento"], x["cambio_pct"]) for x in resumen] == [
        (d, c) for d in ("region", "categoria") for c in (-30, -20, -10, 10, 20, 30)
    ]
    assert len(val["potencia"]["falla_pos"]) == 2 * 3
    assert [s["origen"] for s in r["resumen"]["por_semana"]] == origenes
    with pytest.raises(ValueError, match="semanas de validación"):
        etapa.ejecutar(*rutas, tmp_path / "c.json", semanas_validacion=(date(2023, 3, 19),))
    # No lleva filas ni series: las alertas solo traen segmentos o tiendas
    claves = {k for s in r["semanas"] for al in s["alertas"] for k in al}
    assert claves == set(etapa.CAMPOS)


def test_el_evento_del_monitor_es_el_del_backtest(artefactos_e5):
    # `evento_por_corte` sale de `segmentos` en el fixture: el monitor lo recalcula con el calendario
    a = artefactos_e5
    del_backtest = json.loads(a.metricas.read_text())["protocolo"]["evento_por_corte"]
    cal = monitor.calendario(pl.read_parquet(a.panel))
    assert {o: monitor.evento(cal, date.fromisoformat(o), 7) for o in del_backtest} == del_backtest
    assert sorted(set(del_backtest.values())) == ["buen_fin", "navidad", "post_buen_fin", "resto"]


BINOMIALES = ("tienda_sin_datos", "tasas_globales", "plan_promocion")


def _de(rev: monitor.Revision, campo: str) -> dict:
    """Por medida de calidad y promoción, el `campo` (valor o referencia)."""
    return {
        (m["chequeo"], m["segmento"], m["nivel"], m["medida"]): m[campo]
        for m in rev.medidas
        if m["chequeo"] in BINOMIALES
    }


def _tasa_de_referencia(p: pl.DataFrame) -> float:
    """El oráculo: la tasa sin `units_sold` en los 56 días que terminan en T, sin días especiales."""
    ref = p.filter(pl.col("date").is_between(T - timedelta(days=55), T) & ~monitor.es_especial())
    return ref.select(pl.col("units_sold").is_null().mean()).item()


def test_la_semana_no_entra_a_la_referencia_y_no_incluye_t(limpio):
    p, pron = limpio
    base = monitor.revisar(p, pron, T)
    semana = pl.col("date").is_between(T + timedelta(days=1), T + timedelta(days=7))
    vacia = p.with_columns(
        pl.when(semana).then(None).otherwise(pl.col("units_sold")).alias("units_sold"),
        pl.when(semana).then(1).otherwise(pl.col("has_promotion")).alias("has_promotion"),
    )
    rev = monitor.revisar(vacia, pron, T)
    # Vaciar la semana no mueve la referencia
    assert _de(rev, "referencia") == _de(base, "referencia")
    # Las 42 celdas de cada tienda (del día 1 al 7, sin T) quedan sin datos y todas las filas con promoción
    tiendas = [m for m in rev.medidas if m["chequeo"] == "tienda_sin_datos"]
    assert {m["valor"] for m in tiendas} == {100.0}
    assert {m["valor"] for m in rev.medidas if m["chequeo"] == "plan_promocion"} == {100.0}
    # Cada tienda se compara con la tasa global de la referencia
    assert all(m["referencia"] == pytest.approx(100 * _tasa_de_referencia(p)) for m in tiendas)
    # Sin pares completos no hay desempeño, pero no truena
    assert rev.desempeno == {m: {"wape_pct": None, "sesgo_pct": None, "pares": 0} for m in MODELOS}

    # Vaciar el día T mueve la referencia y no la semana
    solo_t = p.with_columns(
        pl.when(pl.col("date") == T).then(None).otherwise(pl.col("units_sold")).alias("units_sold")
    )
    rev_t = monitor.revisar(solo_t, pron, T)
    assert _de(rev_t, "valor") == _de(base, "valor")
    assert _de(rev_t, "referencia") != _de(base, "referencia")


def test_los_umbrales_aplicados_son_bonferroni_y_la_tienda_es_de_una_cola(limpio):
    p, pron = limpio
    rev = monitor.revisar(inyeccion.falla_pos(p, T, dias=1), pron, T)
    # 20 tiendas, 3 tasas y 19 segmentos
    esperado = {"tienda_sin_datos": 0.01 / 20, "tasas_globales": 0.01 / 3, "plan_promocion": 0.01 / 19}
    aplicados = {c: {m["umbral"] for m in rev.medidas if m["chequeo"] == c} for c in BINOMIALES}
    assert aplicados == {c: {u} for c, u in esperado.items()}
    tasa = _tasa_de_referencia(p)
    for m in rev.medidas:
        if m["chequeo"] == "tienda_sin_datos":
            casos = round(m["valor"] * 42 / 100)
            assert m["p_valor"] == pytest.approx(stats.binom.sf(casos - 1, 42, tasa))  # P(X ≥ casos)
        if m["chequeo"] in esperado:
            assert (m["severidad"] == "alerta") == (m["p_valor"] < esperado[m["chequeo"]])
    # La falla de un día deja a STR_001 con 6 o más celdas vacías: el oráculo cubre casos con k > 0
    falla = next(m for m in rev.medidas if m["chequeo"] == "tienda_sin_datos" and m["nivel"] == "STR_001")
    assert round(falla["valor"] * 42 / 100) >= 6


@pytest.mark.parametrize(
    ("valor", "severidad"), [(0.25, "alerta"), (0.2499, "vigilar"), (0.10, "vigilar"), (0.0999, None)]
)
def test_los_bordes_del_psi(limpio, monkeypatch, valor, severidad):
    p, pron = limpio
    monkeypatch.setattr(monitor.estadisticos, "psi", lambda *args: valor)
    rev = monitor.revisar(p, pron, T)
    psi = [m for m in rev.medidas if m["chequeo"] in ("nivel_demanda", "ticket")]
    assert len(psi) == 2 * 19 and {m["severidad"] for m in psi} == {severidad}


def test_el_ticket_descarta_las_filas_sin_transacciones(limpio):
    p, pron = limpio
    # Filas de una tienda en la referencia y en la semana: sin transacciones (∞) o, en Ropa, también sin
    # monto (NaN)
    sin = (pl.col("store_id") == "STR_002") & (pl.col("date").dt.day() % 3 == 0)
    infinito = p.with_columns(
        pl.when(sin).then(0).otherwise(pl.col("total_transactions")).alias("total_transactions")
    )
    nan = infinito.with_columns(
        pl.when(sin & (pl.col("category") == "Ropa"))
        .then(0.0)
        .otherwise(pl.col("amount_total"))
        .alias("amount_total")
    )
    sin_monto = p.with_columns(
        pl.when(sin).then(None).otherwise(pl.col("amount_total")).alias("amount_total")
    )

    def tickets(panel: pl.DataFrame) -> dict:
        medidas = monitor.revisar(panel, pron, T).medidas
        return {(m["segmento"], m["nivel"]): m["valor"] for m in medidas if m["chequeo"] == "ticket"}

    esperado = tickets(sin_monto)
    assert all(np.isfinite(v) for v in esperado.values())
    assert tickets(infinito) == esperado and tickets(nan) == esperado


def _alerta(chequeo: str, clasificacion: str, nivel: str = "Sur") -> dict:
    capa, cuando, medida = monitor.CHEQUEOS[chequeo]
    return {
        "capa": capa,
        "chequeo": chequeo,
        "segmento": "region",
        "nivel": nivel,
        "modelo": None,
        "medida": medida,
        "severidad": "alerta",
        "clasificacion": clasificacion,
        "cuando": cuando,
    }


def test_informativa_pide_que_todas_las_alertas_sean_esperadas():
    esperadas = [_alerta("nivel_demanda", "esperada", n) for n in ("Sur", "Norte", "Centro")]

    def codigos(alertas: list[dict]) -> list[str]:
        return [r["codigo"] for r in monitor.recomendar(alertas, [])]

    assert codigos(esperadas) == ["informativa"]
    # Una sola inesperada basta para que no sea informativa, aunque no dispare otra recomendación
    assert codigos([*esperadas, _alerta("ticket", "inesperada")]) == ["sin_accion"]
    assert codigos([*esperadas, _alerta("nivel_demanda", "inesperada", "Oriente")]) == ["reentrenar"]


def test_la_banda_es_1_5_por_la_mediana():
    # Mediana 11 y media 17: la banda es 16.5
    def revision(wape: float) -> monitor.Revision:
        return monitor.Revision(
            T, "resto", [], {}, [], {"m": {"wape_pct": wape, "sesgo_pct": 0.0, "pares": 10}}
        )

    revisiones = [revision(10.0), revision(11.0), revision(30.0)]
    assert etapa.bandas(revisiones, monitor.Umbrales()) == {
        "m": {"semanas": 3, "mediana_pct": 11.0, "banda_pct": 16.5}
    }


def test_la_fila_del_corte_k_no_mira_los_cortes_siguientes():
    pron = _pronosticos(ventaja=0.05)
    base = monitor.vigilar_retador(pron, "campeon", "retador", 7)
    k = 8
    despues = pl.col("origen") > date.fromisoformat(base[k - 1]["corte"])
    alterado = pron.with_columns(
        pl.when(despues & (pl.col("modelo") == "retador"))
        .then(pl.col("yhat") * 3)
        .otherwise(pl.col("yhat"))
        .alias("yhat")
    )
    nuevo = monitor.vigilar_retador(alterado, "campeon", "retador", 7)
    assert nuevo[:k] == base[:k]
    assert nuevo[-1]["dm"] != base[-1]["dm"]
