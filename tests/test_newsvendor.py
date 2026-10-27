"""El newsvendor: cuantil ponderado, ventana expansiva, métricas, supuestos y la etapa de punta a punta."""

import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from demanda.decision import etapa, newsvendor
from demanda.decision.supuestos import Supuestos
from demanda.seleccion.diebold_mariano import diebold_mariano

PARES = {
    "modelo": pl.String,
    "origen": pl.Date,
    "store_id": pl.String,
    "category": pl.String,
    "y": pl.Float64,
    "yhat": pl.Float64,
    "normal": pl.Boolean,
}


def test_cuantil_ponderado_a_mano_con_empates_y_peso_cero():
    # En orden: 1 (peso 2), 2 (peso 1; el otro 2 pesa 0 y no entra), 3 y 5 (peso 1). El total es 5 y el
    # acumulado 2, 3, 4, 5: se toma el menor valor cuyo acumulado llega a FC · 5
    valores, pesos = [3, 1, 2, 2, 5], [1, 2, 0, 1, 1]
    assert newsvendor.cuantil_ponderado(valores, pesos, 0.4) == 1  # el acumulado llega justo a 2
    assert newsvendor.cuantil_ponderado(valores, pesos, 0.5) == 2
    assert newsvendor.cuantil_ponderado(valores, pesos, 0.8) == 3
    assert newsvendor.cuantil_ponderado(valores, pesos, 0.81) == 5
    # Los empates suman su peso: los dos 2 cubren del 25 al 75 % del acumulado
    assert newsvendor.cuantil_ponderado([1, 2, 2, 3], [1, 1, 1, 1], 0.75) == 2
    assert newsvendor.cuantil_ponderado([1, 2, 2, 3], [1, 1, 1, 1], 0.76) == 3
    with pytest.raises(ValueError, match="peso positivo"):
        newsvendor.cuantil_ponderado([1, 2], [0, 0], 0.5)


CALIBRACION = [date(2023, 1, 5), date(2023, 1, 12)]
CORTES = [date(2023, 1, 19), date(2023, 1, 26), date(2023, 2, 2), date(2023, 2, 9)]
EVENTOS = {CORTES[0]: "resto", CORTES[1]: "buen_fin", CORTES[2]: "resto", CORTES[3]: "resto"}


def _pares() -> pl.DataFrame:
    """20 series por semana con r uniforme entre 0.8 y 1.2, salvo en el Buen Fin, donde r = 2."""
    rng = np.random.default_rng(0)
    filas = []
    for origen in CALIBRACION + CORTES:
        evento = EVENTOS.get(origen, "resto")
        for i in range(20):
            r = 2.0 if evento == "buen_fin" else rng.uniform(0.8, 1.2)
            filas.append(("m", origen, f"STR_{i:03d}", "Ropa", 100 * r, 100.0, evento == "resto"))
    return pl.DataFrame(filas, schema=PARES, orient="row")


def test_la_ventana_no_mira_el_futuro_ni_usa_semanas_con_evento():
    pares, fc = _pares(), {"Ropa": 0.9}
    base = newsvendor.multiplicadores(pares, CORTES, fc, 7)
    # Las 2 semanas de calibración y los cortes normales cerrados: el del Buen Fin no entra
    assert base["semanas"].to_list() == [2, 3, 3, 4]
    assert base["pares"].to_list() == [40, 60, 60, 80]
    assert base.select(pl.col("q").max()).item() < 1.25

    for k, corte in enumerate(CORTES):
        alterados = pares.with_columns(
            pl.when(pl.col("origen") >= corte).then(pl.col("y") * 5).otherwise(pl.col("y")).alias("y")
        )
        assert newsvendor.multiplicadores(alterados, [corte], fc, 7)["q"].item() == base["q"][k]
    # Con el Buen Fin en la ventana, 20 de 80 pares con r = 2 llevan el cuantil 0.9 a 2
    todos = pares.with_columns(pl.lit(True).alias("normal"))
    assert newsvendor.multiplicadores(todos, [CORTES[2]], fc, 7)["q"].item() == 2.0


def test_los_pares_sin_pronostico_no_calibran():
    pares = _pares().with_columns(
        pl.when(pl.col("store_id") == "STR_000").then(0.0).otherwise(pl.col("yhat")).alias("yhat")
    )
    assert newsvendor.multiplicadores(pares, CORTES[:1], {"Ropa": 0.9}, 7)["pares"].item() == 38


def test_con_fc_de_un_medio_y_errores_simetricos_q_es_casi_1():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.15, 400)
    r = np.concatenate([1 + e, 1 - e])
    filas = [("m", CALIBRACION[0], f"STR_{i:03d}", "Ropa", 50 * x, 50.0, True) for i, x in enumerate(r)]
    pares = pl.DataFrame(filas, schema=PARES, orient="row")
    q = newsvendor.multiplicadores(pares, CORTES[:1], {"Ropa": 0.5}, 7)["q"].item()
    assert q == pytest.approx(1.0, abs=0.01)


def test_solo_entran_los_pares_completos():
    # Tres series de un corte: A con los 7 días, B con un real nulo y C con un día de menos
    filas = []
    for tienda, dias, nulo in (("A", 7, None), ("B", 7, 3), ("C", 6, None)):
        for h in range(1, dias + 1):
            y = None if h == nulo else 10.0 * h
            filas.append(("m", CORTES[0], tienda, "Ropa", h, y, 2.0 * h))
    columnas = {
        "modelo": pl.String,
        "origen": pl.Date,
        "store_id": pl.String,
        "category": pl.String,
        "h": pl.Int64,
        "y": pl.Float64,
        "yhat": pl.Float64,
    }
    pares = newsvendor.pares_semanales(pl.DataFrame(filas, schema=columnas, orient="row"), 7)
    assert pares.select("store_id", "y", "yhat").rows() == [("A", 280.0, 56.0)]


def test_las_metricas_de_una_politica_a_mano():
    o = CORTES[0]
    niveles = pl.DataFrame(
        {
            "politica": ["p", "p"],
            "origen": [o, o],
            "store_id": ["STR_001", "STR_002"],
            "category": ["Ropa", "Ropa"],
            "y": [10.0, 10.0],
            "S": [12.0, 7.0],
        }
    )
    costos = pl.DataFrame({"category": ["Ropa"], "cu": [2.0], "co": [0.5], "fc": [0.8]})
    res = newsvendor.resultados(niveles, costos)
    # Sobra 2 en el primero (0.5 · 2 = 1 peso) y faltan 3 en el segundo (2 · 3 = 6 pesos)
    assert res["sobre"].to_list() == [2.0, 0.0] and res["quiebre"].to_list() == [0.0, 3.0]
    assert res["costo"].to_list() == [1.0, 6.0]
    m = newsvendor.metricas(res, ["politica"]).row(0, named=True)
    assert (m["pares"], m["demanda"], m["stock"], m["costo"]) == (2, 20.0, 19.0, 7.0)
    assert m["sobreinventario_pct"] == pytest.approx(10.0)
    assert m["quiebre_pct"] == pytest.approx(15.0)
    assert m["fill_rate_pct"] == pytest.approx(85.0)
    assert m["nivel_servicio_pct"] == pytest.approx(50.0)
    assert m["fc_objetivo_pct"] == pytest.approx(80.0)
    assert m["stock_sobre_demanda"] == pytest.approx(0.95)
    assert newsvendor.costo_por_corte(res)["costo"].to_list() == [7.0]


def test_supuestos_y_costos_unitarios():
    s = Supuestos()
    fc = {c: round(v, 3) for c, v in s.fracciones().items()}
    assert fc == {
        "Abarrotes": 0.909,
        "Bebidas": 0.943,
        "Cuidado_Personal": 0.952,
        "Electronica": 0.857,
        "Hogar": 0.938,
        "Ropa": 0.918,
    }
    precios = dict.fromkeys(s.costos, 100.0)
    base = newsvendor.costos_unitarios(precios, s).filter(pl.col("category") == "Ropa").row(0, named=True)
    assert (base["cu"], base["co"]) == pytest.approx((45.0, 4.0))
    # La sensibilidad mantiene Cu y despeja Co de la fracción crítica
    sens = (
        newsvendor.costos_unitarios(precios, s, 0.8).filter(pl.col("category") == "Ropa").row(0, named=True)
    )
    assert (sens["cu"], sens["co"], sens["fc"]) == pytest.approx((45.0, 11.25, 0.8))
    assert sens["cu"] / (sens["cu"] + sens["co"]) == pytest.approx(0.8)
    with pytest.raises(AttributeError):
        s.fracciones_sensibilidad = ()  # pyrefly: ignore[read-only]


def test_el_precio_medio_usa_la_historia_hasta_el_corte():
    p = pl.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            "category": ["Ropa"] * 4,
            "amount_total": [100.0, 300.0, 50.0, 1000.0],
            "units_sold": [1, 2, None, 1],
        }
    )
    # Hasta el 3: (100 + 300) / (1 + 2); la fila sin unidades no entra y el 4 queda fuera
    assert newsvendor.precio_medio(p, date(2023, 1, 3)) == {"Ropa": pytest.approx(400 / 3)}


def test_decision_de_punta_a_punta_es_reproducible(artefactos_e5, tmp_path):
    a = artefactos_e5
    rutas = (a.pronosticos, a.metricas, a.seleccion, a.calibracion, a.panel)
    r = etapa.ejecutar(*rutas, tmp_path / "a.json")
    etapa.ejecutar(*rutas, tmp_path / "b.json")
    texto = (tmp_path / "a.json").read_text()
    assert texto == (tmp_path / "b.json").read_text()
    assert json.loads(texto) == json.loads(json.dumps(r))
    assert set(r["fuente"]) == {
        "pronosticos_sha256",
        "metricas_sha256",
        "seleccion_sha256",
        "calibracion_sha256",
        "panel_sha256",
    }

    campeon, retador = r["campeon"], r["retador"]
    politicas = [f"puntual_{campeon}", f"newsvendor_{campeon}", f"newsvendor_{retador}"]
    assert list(r["politicas"]) == politicas
    niveles = {(f["segmento"], f["nivel"]) for f in r["impacto"]}
    assert {
        ("global", "global"),
        ("evento", "resto"),
        ("formato", "Bodega"),
        ("categoria", "Ropa"),
    } <= niveles
    globales = {f["politica"]: f for f in r["impacto"] if f["segmento"] == "global"}
    assert list(globales) == politicas
    for f in globales.values():
        assert f["fill_rate_pct"] == pytest.approx(100 - f["quiebre_pct"], abs=1e-3)
        assert f["pares"] == globales[politicas[0]]["pares"]

    # El puntual no lleva colchón: su stock es la suma del pronóstico del campeón en los pares completos
    pronosticos = pl.read_parquet(a.pronosticos).filter(pl.col("modelo") == campeon)
    pares = newsvendor.pares_semanales(pronosticos, 7)
    assert globales[politicas[0]]["stock"] == pytest.approx(pares["yhat"].sum(), abs=0.01)
    assert globales[politicas[0]]["pares"] == pares.height

    # 4 cortes × 2 modelos × 6 categorías. La ventana ve las 2 semanas de calibración y el corte normal del
    # 03-25; el Buen Fin (04-01), la semana posterior (04-08) y la Navidad (04-15) no entran
    eventos = json.loads(a.metricas.read_text())["protocolo"]["evento_por_corte"]
    assert eventos == {
        "2023-03-25": "resto",
        "2023-04-01": "buen_fin",
        "2023-04-08": "post_buen_fin",
        "2023-04-15": "navidad",
    }
    assert len(r["multiplicadores"]) == 4 * 2 * 6
    semanas = {c: {m["semanas"] for m in r["multiplicadores"] if m["corte"] == c} for c in eventos}
    assert semanas == {"2023-03-25": {2}, "2023-04-01": {3}, "2023-04-08": {3}, "2023-04-15": {3}}

    pruebas = [(p["a"], p["b"], p["segmento"]) for p in r["pruebas_costo"]]
    assert pruebas == [
        (politicas[2], politicas[1], "global"),
        (politicas[2], politicas[1], "resto"),
        (politicas[1], politicas[0], "global"),
        (politicas[1], politicas[0], "resto"),
    ]
    costo = {(f["segmento"], f["nivel"], f["politica"]): f["costo"] for f in r["impacto"]}
    normales = len(r["calibracion"]["cortes_normales"])
    for p in r["pruebas_costo"]:
        nivel = ("global", "global") if p["segmento"] == "global" else ("evento", "resto")
        assert p["costo_a"] == pytest.approx(costo[(nivel[0], nivel[1], p["a"])], abs=0.02)
        assert p["costo_b"] == pytest.approx(costo[(nivel[0], nivel[1], p["b"])], abs=0.02)
        assert p["diferencia"] == pytest.approx(p["costo_a"] - p["costo_b"], abs=0.02)
        assert p["n_cortes"] == (4 if p["segmento"] == "global" else normales)
        # Con un solo corte normal no hay prueba en `resto`
        assert (p["p_valor"] is None) == (p["segmento"] == "resto")

    sens = r["sensibilidad"]
    assert [s["fraccion_critica"] for s in sens["fracciones"]] == [0.8, 0.9, 0.95, 0.98]
    assert all(sorted(s["orden"]) == sorted(politicas) for s in sens["fracciones"])
    assert sens["orden_se_mantiene"] == all(s["orden"] == sens["orden_base"] for s in sens["fracciones"])
    # El puntual no depende de la fracción crítica: su fill rate es el mismo en toda la sensibilidad
    assert len({s["global"][politicas[0]]["fill_rate_pct"] for s in sens["fracciones"]}) == 1
    # La sensibilidad recalcula los multiplicadores: suben con la fracción crítica
    for modelo in (campeon, retador):
        medios = [s["multiplicador_medio"][modelo] for s in sens["fracciones"]]
        assert medios == sorted(medios) and medios[0] < medios[-1]
    assert campeon in r["conclusion"] and retador in r["conclusion"]


def test_la_dm_sobre_el_costo_es_la_de_e4(artefactos_e5, tmp_path):
    a = artefactos_e5
    r = etapa.ejecutar(a.pronosticos, a.metricas, a.seleccion, a.calibracion, a.panel, tmp_path / "i.json")
    # Se rehace el costo por corte con las mismas piezas y se compara con una llamada directa
    seleccion = json.loads(a.seleccion.read_text())
    eventos = {
        date.fromisoformat(c): e
        for c, e in json.loads(a.metricas.read_text())["protocolo"]["evento_por_corte"].items()
    }
    backtest = pl.read_parquet(a.pronosticos)
    calib = pl.read_parquet(a.calibracion)
    precios = newsvendor.precio_medio(pl.read_parquet(a.panel), min(eventos))
    costos = newsvendor.costos_unitarios(precios, Supuestos())
    _, res = etapa.evaluar(
        newsvendor.pares_semanales(calib, 7),
        newsvendor.pares_semanales(backtest.filter(pl.col("modelo").is_in([r["campeon"], r["retador"]])), 7),
        eventos,
        r["campeon"],
        r["retador"],
        costos,
        7,
    )
    por_corte = (
        newsvendor.costo_por_corte(res).pivot("politica", index="origen", values="costo").sort("origen")
    )
    a_, b_ = f"newsvendor_{r['campeon']}", f"puntual_{r['campeon']}"
    directa = diebold_mariano((por_corte[a_] - por_corte[b_]).to_list())
    prueba = next(p for p in r["pruebas_costo"] if (p["a"], p["b"], p["segmento"]) == (a_, b_, "global"))
    assert prueba["p_valor"] == pytest.approx(directa.p_valor, abs=1e-6)
    assert prueba["a_favor_de"] == (a_ if directa.media < 0 else b_)
    assert seleccion["campeon"] == r["campeon"]


def test_decision_rechaza_una_seleccion_de_otra_corrida(artefactos_e5, tmp_path):
    a = artefactos_e5
    seleccion = json.loads(a.seleccion.read_text())
    seleccion["fuente"]["pronosticos_sha256"] = "otro"
    otra = tmp_path / "seleccion.json"
    otra.write_text(json.dumps(seleccion))
    with pytest.raises(ValueError, match="corre `demanda seleccion`"):
        etapa.ejecutar(a.pronosticos, a.metricas, otra, a.calibracion, a.panel, tmp_path / "i.json")


def _res(costos_a: list[float], costos_b: list[float]) -> pl.DataFrame:
    """Un par por política y corte con el costo dado, para probar las comparaciones."""
    filas = [
        (politica, CORTES[k], "STR_001", "Ropa", costo)
        for politica, costos in (("a", costos_a), ("b", costos_b))
        for k, costo in enumerate(costos)
    ]
    columnas = ["politica", "origen", "store_id", "category", "costo"]
    return pl.DataFrame(filas, schema=columnas, orient="row")


def test_una_diferencia_constante_o_un_costo_cero_dan_un_resultado_definido():
    # d_t = 5 en todos los cortes: la DM no está definida y queda en nulo, sin excepción
    constante = etapa._pruebas(_res([15.0, 25.0, 35.0], [10.0, 20.0, 30.0]), [("a", "b")], CORTES[:3])
    assert [(p["segmento"], p["n_cortes"], p["p_valor"]) for p in constante] == [
        ("global", 3, None),
        ("resto", 3, None),
    ]
    assert constante[0]["diferencia"] == 15.0 and constante[0]["diferencia_pct"] == 25.0
    # Sin costo en b, la diferencia relativa queda en nulo y la conclusión da la diferencia en pesos
    sin_costo = etapa._pruebas(_res([1.0, 2.0, 4.0], [0.0, 0.0, 0.0]), [("a", "b")], [])
    assert sin_costo[0]["diferencia_pct"] is None and sin_costo[0]["p_valor"] is not None
    texto = etapa._conclusion([sin_costo[0], {**constante[0], "diferencia": 0.0}], True, 4)
    assert "7.00 pesos más que b" in texto and "lo mismo que b" in texto
