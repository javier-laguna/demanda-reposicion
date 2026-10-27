"""La regla de campeón: un caso por criterio, el desempate, la sensibilidad y la salida de punta a punta."""

import json
from datetime import date

import polars as pl
import pytest
from scipy import stats

from demanda.backtest import bucle, segmentos
from demanda.backtest.protocolo import Protocolo
from demanda.datos.contratos import SERIE
from demanda.datos.fuente import sha256
from demanda.modelos import Contexto, crear
from demanda.seleccion import campeon
from demanda.seleccion.campeon import Evidencia, Regla, criterios, elegir, sensibilidad
from demanda.seleccion.diebold_mariano import PruebaDM

REGLA = Regla()
INCUMBENTE = Evidencia(27.3, 11.0, 2.4, "global=global", None)


def _dm(p: float = 0.01, media: float = -50.0) -> PruebaDM:
    return PruebaDM(17, media, -3.0 if media < 0 else 3.0, p)


def _evidencias(**cambios) -> dict[str, Evidencia]:
    """El incumbente, un baseline que pierde en semanas normales y un retador que lo cumple todo salvo
    `cambios`."""
    retador = Evidencia(24.0, 11.2, 2.0, "formato=Bodega", _dm())
    return {
        "sistema_actual": INCUMBENTE,
        "autoets": Evidencia(24.7, 13.4, 3.1, "global=global", _dm(p=0.3)),
        "lightgbm": retador._replace(**cambios),
    }


def _fallan(ev: dict[str, Evidencia], modelo: str = "lightgbm") -> list[str]:
    evaluados = criterios(ev[modelo], INCUMBENTE, REGLA, REGLA.margen_resto_pp, REGLA.alfa)
    return [n for n, c in evaluados.items() if not c.cumple]


def test_un_retador_que_cumple_las_cuatro_condiciones_es_campeon():
    ev = _evidencias()
    assert _fallan(ev) == []
    assert _fallan(ev, "autoets") == ["significativa", "no_pierde_en_resto"]
    assert elegir(ev, REGLA) == ("lightgbm", ["lightgbm"])


@pytest.mark.parametrize(
    ("criterio", "cambios"),
    [
        ("gana_global", {"wape_global": 27.5}),
        ("gana_global", {"wape_global": 27.3}),  # empatar no es ganar
        ("significativa", {"dm_global": _dm(p=0.2)}),
        ("significativa", {"dm_global": _dm(p=0.05)}),  # p < α es estricto
        ("significativa", {"dm_global": _dm(p=0.01, media=50.0)}),  # significativa, pero en contra
        ("no_pierde_en_resto", {"wape_resto": 11.6}),
        ("sin_sesgo", {"sesgo_max": 5.3, "sesgo_peor": "region=Sur"}),
    ],
)
def test_si_falla_un_solo_criterio_sigue_el_incumbente(criterio, cambios):
    ev = _evidencias(**cambios)
    assert _fallan(ev) == [criterio]
    assert elegir(ev, REGLA) == ("sistema_actual", [])


def test_los_umbrales_de_resto_y_sesgo_son_inclusivos():
    ev = _evidencias(wape_resto=11.5, sesgo_max=5.0)
    assert _fallan(ev) == []


def test_desempate_por_wape_global_y_luego_por_orden_del_registro():
    ev = _evidencias()
    ev["chronos2"] = ev["lightgbm"]._replace(wape_global=23.0)
    assert elegir(ev, REGLA) == ("chronos2", ["lightgbm", "chronos2"])
    ev["chronos2"] = ev["lightgbm"]
    assert elegir(ev, REGLA) == ("lightgbm", ["lightgbm", "chronos2"])


def test_sensibilidad_al_margen_y_a_alfa():
    # Pasa solo con margen ≥ 1 pp (11.8 contra 11.0) y α = 0.10 (p = 0.07)
    ev = _evidencias(wape_resto=11.8, dm_global=_dm(p=0.07))
    filas = {(f["margen_resto_pp"], f["alfa"]): (f["pasan"], f["campeon"]) for f in sensibilidad(ev, REGLA)}

    assert len(filas) == 8
    pasa = {(1.0, 0.10), (2.0, 0.10)}
    for clave, (pasan, ganador) in filas.items():
        assert pasan == (["lightgbm"] if clave in pasa else []), clave
        assert ganador == ("lightgbm" if clave in pasa else "sistema_actual")
    # La sensibilidad no cambia la regla
    assert elegir(ev, REGLA)[0] == "sistema_actual"


def test_la_regla_es_inmutable():
    with pytest.raises(AttributeError):
        REGLA.alfa = 0.10  # pyrefly: ignore[read-only]


BUEN_FIN = date(
    2023, 2, 20
)  # en el horizonte del corte del 2023-02-18; el del 02-25 queda como post_buen_fin
NORMALES = ["2023-01-28", "2023-02-04", "2023-02-11", "2023-03-04"]


class Sesgado:
    """Pronostica el real × 0.8 en STR_001 y × 0.97 en las demás tiendas. Cada formato, región y NSE tiene una
    sola tienda, así que su sesgo semanal es −20 % en los niveles de STR_001 y −3 % en los demás."""

    def __init__(self, panel: pl.DataFrame) -> None:
        self.reales = panel.select(*SERIE, "date", pl.col("units_sold").cast(pl.Float64).alias("y"))

    def pronosticar(self, historia: pl.DataFrame, futuro: pl.DataFrame) -> pl.DataFrame:
        factor = pl.when(pl.col("store_id") == "STR_001").then(0.8).otherwise(0.97)
        return (
            futuro.select(*SERIE, "date", "h")
            .join(self.reales, on=[*SERIE, "date"], how="left")
            .select(*SERIE, "date", "h", (pl.col("y").fill_null(0) * factor).alias("yhat"))
        )


@pytest.fixture(scope="module")
def salidas(panel_sintetico, tmp_path_factory):
    """Las salidas de un backtest sintético de 6 cortes, con un Buen Fin, tres baselines y `Sesgado`."""
    panel = panel_sintetico.with_columns((pl.col("date") == BUEN_FIN).alias("is_buen_fin"))
    protocolo = Protocolo(n_cortes=6)
    modelos = {
        **crear(Contexto(factor_senal=0.53), ["sistema_actual", "naive_estacional", "ma7"]),
        "sesgado": Sesgado(panel),
    }
    resultado = bucle.correr(panel, modelos, protocolo)
    metricas = {
        "protocolo": {
            "horizonte": protocolo.horizonte,
            "origenes": [str(o) for o in protocolo.origenes(panel["date"].max())],
            "modelos": list(modelos),
            "evento_por_corte": segmentos.evento_por_corte(resultado, panel, protocolo.horizonte),
        },
        "segmentos": segmentos.tabla(resultado, panel, protocolo, list(modelos)).to_dicts(),
    }
    carpeta = tmp_path_factory.mktemp("backtest")
    (carpeta / "metricas.json").write_text(json.dumps(metricas))
    resultado.pronosticos.write_parquet(carpeta / "pronosticos.parquet")
    return carpeta, metricas, resultado.pronosticos


@pytest.fixture(scope="module")
def seleccion(salidas, tmp_path_factory):
    carpeta, _, _ = salidas
    salida = tmp_path_factory.mktemp("seleccion") / "seleccion.json"
    return campeon.ejecutar(carpeta / "metricas.json", carpeta / "pronosticos.parquet", salida)


def _por_par(pronosticos: pl.DataFrame) -> pl.DataFrame:
    """Suma semanal por modelo × corte × serie, solo en los pares con los 7 días observados."""
    return (
        pronosticos.group_by("modelo", "origen", *SERIE)
        .agg(pl.col("y").sum(), pl.col("yhat").sum(), pl.col("y").is_not_null().all().alias("ok"), pl.len())
        .filter(pl.col("ok") & (pl.col("len") == 7))
        .with_columns(pl.col("origen").cast(pl.String))
    )


def test_los_eventos_del_fixture(salidas):
    _, metricas, _ = salidas
    eventos = metricas["protocolo"]["evento_por_corte"]
    assert eventos["2023-02-18"] == "buen_fin" and eventos["2023-02-25"] == "post_buen_fin"
    assert sorted(c for c, e in eventos.items() if e == "resto") == NORMALES


def test_el_wape_de_resto_es_el_de_las_semanas_normales(salidas, seleccion):
    _, _, pronosticos = salidas
    pares = _por_par(pronosticos)

    def wape(df: pl.DataFrame) -> dict[str, float]:
        agregado = df.group_by("modelo").agg(
            (100 * (pl.col("yhat") - pl.col("y")).abs().sum() / pl.col("y").sum()).alias("w")
        )
        return dict(agregado.iter_rows())

    en_resto, en_global = wape(pares.filter(pl.col("origen").is_in(NORMALES))), wape(pares)
    for modelo, m in seleccion["modelos"].items():
        assert m["wape_semanal_resto_pct"] == pytest.approx(en_resto[modelo], abs=1e-3)
        assert m["wape_semanal_global_pct"] == pytest.approx(en_global[modelo], abs=1e-3)
    # Los cortes del Buen Fin y del posterior separan resto del global: leer uno por otro no pasaría
    assert max(abs(en_resto[m] - en_global[m]) for m in en_resto) > 0.5


def test_el_peor_sesgo_es_el_mayor_en_valor_absoluto(seleccion):
    sesgado = seleccion["modelos"]["sesgado"]
    # −20 % en Bodega, Centro y A/B (STR_001) contra −3 % en los demás niveles: el peor es negativo y, con el
    # empate, el primer segmento en orden alfabético
    assert sesgado["sesgo_semanal_max_abs_pct"] == pytest.approx(20.0, abs=1e-3)
    assert sesgado["sesgo_semanal_peor_subgrupo"] == "formato=Bodega"
    assert sesgado["criterios"]["sin_sesgo"] == {"valor": 20.0, "umbral": 5.0, "cumple": False}


def test_la_dm_de_resto_usa_solo_los_cortes_normales(salidas, seleccion):
    _, _, pronosticos = salidas
    pares = _por_par(pronosticos).filter(pl.col("origen").is_in(NORMALES))
    abs_por_corte = pares.group_by("modelo", "origen").agg(
        (pl.col("yhat") - pl.col("y")).abs().sum().alias("e")
    )
    tabla = abs_por_corte.pivot("modelo", index="origen", values="e").sort("origen")
    d = (tabla["ma7"] - tabla["sistema_actual"]).to_list()
    t = stats.ttest_1samp(d, 0)  # DM con la corrección HLN y h = 1 es la t pareada

    prueba = next(
        p
        for p in seleccion["pruebas_dm"]
        if (p["a"], p["b"], p["segmento"]) == ("ma7", "sistema_actual", "resto")
    )
    assert prueba["n_cortes"] == len(NORMALES)
    assert prueba["p_valor"] == pytest.approx(t.pvalue, abs=1e-6)
    assert prueba["estadistico"] == pytest.approx(t.statistic, abs=1e-4)


def test_seleccion_de_punta_a_punta_es_reproducible(salidas, seleccion, tmp_path):
    carpeta, metricas, _ = salidas
    rutas = (carpeta / "metricas.json", carpeta / "pronosticos.parquet")
    campeon.ejecutar(*rutas, tmp_path / "a.json")
    campeon.ejecutar(*rutas, tmp_path / "b.json")
    s = seleccion

    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
    assert json.loads((tmp_path / "a.json").read_text()) == json.loads(json.dumps(s))
    assert s["fuente"] == {"metricas_sha256": sha256(rutas[0]), "pronosticos_sha256": sha256(rutas[1])}
    assert s["regla"]["alfa"] == 0.05 and s["regla"]["margen_resto_pp"] == 0.5
    assert s["incumbente"] == "sistema_actual"
    assert s["mejor_baseline"]["modelo"] in ("naive_estacional", "ma7")
    assert s["cortes"] == {"total": 6, "resto": 4}
    assert s["campeon"] in metricas["protocolo"]["modelos"] and s["justificacion"]
    assert len(s["sensibilidad"]) == 8
    assert s["modelos"]["sistema_actual"]["criterios"] is None

    # Tres retadores contra el incumbente y `sesgado` (que no es baseline) contra el mejor baseline
    pruebas = {(p["a"], p["b"], p["segmento"]): p for p in s["pruebas_dm"]}
    retadores = ("naive_estacional", "ma7", "sesgado")
    esperadas = {(m, "sistema_actual", seg) for m in retadores for seg in ("global", "resto")}
    esperadas |= {("sesgado", s["mejor_baseline"]["modelo"], seg) for seg in ("global", "resto")}
    assert set(pruebas) == esperadas
    # La media de d_t contrasta la diferencia de WAPE semanal global
    wape = {m: v["wape_semanal_global_pct"] for m, v in s["modelos"].items()}
    global_ma7 = pruebas[("ma7", "sistema_actual", "global")]
    assert global_ma7["diferencia_wape_pp"] == pytest.approx(wape["ma7"] - wape["sistema_actual"], abs=1e-3)
    assert global_ma7["n_cortes"] == 6
    assert global_ma7["a_favor_de"] == ("ma7" if global_ma7["media_d"] < 0 else "sistema_actual")


def test_con_menos_de_tres_semanas_normales_la_dm_de_resto_queda_en_nulo(salidas):
    _, metricas, pronosticos = salidas
    eventos = dict.fromkeys(metricas["protocolo"]["evento_por_corte"], "navidad")
    eventos |= dict.fromkeys(NORMALES[:2], "resto")
    s = campeon.seleccionar(
        {**metricas, "protocolo": {**metricas["protocolo"], "evento_por_corte": eventos}}, pronosticos, REGLA
    )

    resto = [p for p in s["pruebas_dm"] if p["segmento"] == "resto"]
    assert resto and all(p["n_cortes"] == 2 for p in resto)
    assert all(p["p_valor"] is None and p["estadistico"] is None and p["a_favor_de"] is None for p in resto)
    # La regla solo usa la DM global, que sigue completa
    assert all(p["p_valor"] is not None for p in s["pruebas_dm"] if p["segmento"] == "global")


def _con_valor(metricas: dict, modelo: str, valor: float) -> dict:
    """Una copia de `metricas` con el WAPE semanal global de `modelo` cambiado."""
    alterada = json.loads(json.dumps(metricas))
    for fila in alterada["segmentos"]:
        clave = (fila["granularidad"], fila["segmento"], fila["modelo"], fila["metrica"])
        if clave == ("semanal", "global", modelo, "wape_pct"):
            fila["valor"] = valor
    return alterada


def test_seleccion_rechaza_artefactos_inconsistentes(salidas):
    _, metricas, pronosticos = salidas
    reportada = next(
        f["valor"]
        for f in metricas["segmentos"]
        if (f["granularidad"], f["segmento"], f["modelo"], f["metrica"])
        == ("semanal", "global", "ma7", "wape_pct")
    )
    protocolo = metricas["protocolo"]
    casos = {
        "WAPE distinta": (_con_valor(metricas, "ma7", reportada + 1), pronosticos),
        "WAPE nula": (_con_valor(metricas, "ma7", float("nan")), pronosticos),
        "un modelo de menos en los pronósticos": (metricas, pronosticos.filter(pl.col("modelo") != "ma7")),
        "un corte de menos en los pronósticos": (
            metricas,
            pronosticos.filter(pl.col("origen") != pl.col("origen").max()),
        ),
        "eventos de otros cortes": (
            {**metricas, "protocolo": {**protocolo, "evento_por_corte": {"2020-01-02": "resto"}}},
            pronosticos,
        ),
    }
    for caso, (m, p) in casos.items():
        with pytest.raises(ValueError, match="misma corrida"):
            campeon.seleccionar(m, p, REGLA)
        assert caso

    sin_eventos = {**metricas, "protocolo": {**protocolo, "evento_por_corte": {}}}
    with pytest.raises(ValueError, match="evento_por_corte"):
        campeon.seleccionar(sin_eventos, pronosticos, REGLA)
