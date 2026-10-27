"""El experimento de promociones con efectos conocidos: balance, efecto, heterogeneidad con Holm, dinámica,
efectos cruzados, aleatorización, potencia y la etapa de punta a punta."""

import json
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from demanda import cli
from demanda.causal import balance, efectos, etapa, potencia
from demanda.causal.absorcion import (
    TIENDA,
    Estimacion,
    absorber,
    codigos,
    coeficiente,
    densos,
    estimar,
    wald,
)
from demanda.causal.efectos import PROMO, REZAGOS, Y, adelanto, rezago
from demanda.causal.potencia import Parametros
from demanda.config import Config
from demanda.datos import panel
from demanda.datos.contratos import CATEGORIAS, SERIE

INICIO = date(2023, 1, 1)
RAPIDOS = Parametros(permutaciones=40, placebos_aa=40, placebos_semanal=20, placebos_por_ventana=10)


def _panel(
    generar_tablas,
    promo: Callable[[pl.DataFrame, np.random.Generator], np.ndarray] | None = None,
    efecto: pl.Expr | None = None,
    n_tiendas: int = 20,
    dias: int = 84,
    ruido: float = 0.1,
    semilla: int = 0,
    quitar: pl.Expr | None = None,
) -> pl.DataFrame:
    """Un panel de la forma real con log(unidades) = α_serie + γ_fecha + efecto + ε.

    - El calendario tiene Buen Fin, Semana Santa y temporada navideña en algunos días, para que B1 y B2
      tengan con qué variar.
    - `promo` recibe el panel con las unidades sin efecto (`base`) y devuelve la asignación; sin ella queda la
      del generador, al azar al 20 %. `efecto` es una expresión que se suma al logaritmo y puede usar la
      promoción.
    - Los nulos de `units_sold` del generador se conservan. `quitar` saca filas de `transactions`: quedan como
      huecos de la malla, con la promoción nula.
    """
    tablas = generar_tablas(n_tiendas=n_tiendas, categorias=CATEGORIAS, dias=dias, semilla=semilla)
    if quitar is not None:
        tablas = tablas._replace(transacciones=tablas.transacciones.filter(~quitar))
    p = panel.construir(tablas).sort([*SERIE, "date"])
    rng = np.random.default_rng(semilla + 1)
    dia = (pl.col("date") - pl.lit(INICIO)).dt.total_days()
    fechas = p["date"].unique().sort()
    gamma = dict(zip(fechas.to_list(), rng.normal(0, 0.3, fechas.len()), strict=True))
    series = p.select(SERIE).unique().sort(SERIE)
    alfa = series.with_columns(pl.Series("alfa", rng.normal(6, 0.5, series.height)))
    p = (
        p.join(alfa, on=SERIE, how="left", maintain_order="left")
        .with_columns(
            dia.is_between(40, 43).alias("is_buen_fin"),
            dia.is_between(60, 61).alias("is_semana_santa"),
            dia.is_between(70, 80).alias("is_navidad_season"),
            (pl.col("alfa") + pl.col("date").replace_strict(gamma, return_dtype=pl.Float64)).alias("base"),
            pl.Series("ruido", rng.normal(0, ruido, p.height)),
        )
        .sort([*SERIE, "date"])
    )
    if promo is not None:
        asignada = pl.Series(PROMO, promo(p, rng)).cast(pl.Int64)
        p = p.with_columns(pl.when(~pl.col("flag_fila_faltante")).then(asignada).alias(PROMO))
    log_y = pl.col("base") + pl.col("ruido") + (pl.lit(0.0) if efecto is None else efecto)
    return p.with_columns(
        pl.when(pl.col("units_sold").is_not_null())
        .then(log_y.exp().round().cast(pl.Int64))
        .alias("units_sold")
    ).drop("alfa", "base", "ruido")


def _evaluar_balance(p: pl.DataFrame) -> dict:
    preparado = efectos.preparar(p)
    m = efectos.muestra(preparado)
    return balance.evaluar(preparado, m, [rezago(1), rezago(7)], [adelanto(k) for k in REZAGOS], Y)


def test_preparar_usa_la_malla_y_la_muestra_excluye_nulos(generar_tablas):
    p = _panel(generar_tablas, n_tiendas=4, dias=21)
    preparado = efectos.preparar(p)
    serie = preparado.filter((pl.col("store_id") == "STR_001") & (pl.col("category") == "Bebidas"))
    promos = serie[PROMO].to_list()
    assert serie[rezago(1)].to_list() == [None, *promos[:-1]]
    assert serie[adelanto(7)].to_list() == [*promos[7:], *[None] * 7]
    # Las otras categorías en promoción el mismo día, sin contar la propia
    dia = preparado.filter((pl.col("store_id") == "STR_001") & (pl.col("date") == INICIO))
    total = int(dia[PROMO].sum())
    assert (dia["otras_en_promocion"] == total - dia[PROMO]).all()
    m = efectos.muestra(preparado)
    assert m.height == p.filter(pl.col("units_sold").is_not_null()).height
    assert m[Y].to_list() == pytest.approx(np.log(m["units_sold"].to_numpy()).tolist())


def test_balance_con_asignacion_al_azar_no_rechaza(generar_tablas):
    b = _evaluar_balance(_panel(generar_tablas, n_tiendas=40))
    assert b["se_sostiene"], b["fallan"]
    assert b["familia"]["n_pruebas"] == 17
    assert [f["prueba"] for f in b["familia"]["pruebas"][:2]] == ["B2:serie", "B2:tienda"]
    # La Wald corregida lleva G − q gl en el denominador: 40 − 27 y 40 − 7
    assert b["b1"]["gl"] == [27, 13] and b["b3"]["gl"] == [7, 33]
    assert b["b1"]["p_valor"] > b["b1"]["p_sin_correccion"]
    # Holm nunca baja un p y conserva el orden
    pruebas = sorted(b["familia"]["pruebas"], key=lambda f: f["p_valor"])
    assert all(f["p_holm"] >= f["p_valor"] for f in pruebas)
    assert [f["p_holm"] for f in pruebas] == sorted(f["p_holm"] for f in pruebas)


def test_una_bandera_que_mueve_la_promocion_no_tumba_la_asignacion(generar_tablas):
    # Como en los datos reales: la bandera marca justo los días con promoción (filas con efectivo)
    p = _panel(generar_tablas, n_tiendas=40)
    p = p.with_columns(
        ((pl.col(PROMO) == 1) & pl.col("cash_transactions").is_not_null()).alias("flag_conteo_inconsistente")
    )
    b = _evaluar_balance(p)
    assert b["fallan"] == ["B5:conteo_inconsistente"] and not b["se_sostiene"]
    assert b["asignacion_se_sostiene"]


def test_b3_detecta_promociones_tras_caidas_de_ventas(generar_tablas):
    def tras_caidas(p: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
        # Más promoción el día siguiente a una venta baja de la serie
        baja = p.select(
            (pl.col("base") + pl.col("ruido") < (pl.col("base") + pl.col("ruido")).median().over(SERIE))
            .shift(1)
            .over(SERIE)
            .fill_null(False)
        ).to_series()
        return (rng.random(p.height) < np.where(baja.to_numpy(), 0.40, 0.05)).astype(int)

    b = _evaluar_balance(_panel(generar_tablas, tras_caidas, n_tiendas=40))
    assert not b["b3"]["cumple"] and b["b3"]["p_valor"] < 1e-6
    assert "B3" in b["fallan"] and not b["se_sostiene"] and not b["asignacion_se_sostiene"]


def test_b2_marca_con_holm_una_tasa_que_depende_del_formato(generar_tablas):
    def por_formato(p: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
        tasa = np.where(p["store_format"].to_numpy() == "Bodega", 0.10, 0.30)
        return (rng.random(p.height) < tasa).astype(int)

    b = _evaluar_balance(_panel(generar_tablas, por_formato, n_tiendas=40))
    formato = next(f for f in b["familia"]["pruebas"] if f["prueba"] == "B2:formato")
    assert formato["p_holm"] < 0.05
    assert "B2:formato" in b["fallan"] and not b["se_sostiene"]


def test_efecto_conocido_en_una_categoria(generar_tablas):
    verdadero = math.log(1.05)
    en_bebidas = (pl.col(PROMO) == 1) & (pl.col("category") == "Bebidas")
    p = _panel(generar_tablas, efecto=pl.when(en_bebidas).then(verdadero).otherwise(0.0), dias=112)
    m = efectos.muestra(efectos.preparar(p))
    het = efectos.heterogeneidad(m)
    bebidas = het["categoria"]["grupos"]["Bebidas"]
    assert bebidas["ic95"][0] < verdadero < bebidas["ic95"][1]
    assert bebidas["ic95_pct"][0] < 5 < bebidas["ic95_pct"][1]
    assert het["familia_holm"]["pruebas"] == 9
    assert het["familia_holm"]["significativos"] == ["categoria:Bebidas"]
    assert het["categoria"]["igualdad"]["hay_heterogeneidad"]
    # El IC en % es la transformación de los extremos del IC en log
    assert bebidas["ic95_pct"] == pytest.approx([100 * math.expm1(b) for b in bebidas["ic95"]])
    # Sin efectos fijos, β es la diferencia de medias del log
    rob = efectos.robustez(m)
    medias = m.group_by(PROMO).agg(pl.col(Y).mean()).sort(PROMO)[Y].to_list()
    assert rob["sin_efectos_fijos"]["beta"] == pytest.approx(medias[1] - medias[0], abs=1e-12)
    assert rob["solo_dias_normales"]["n"] < m.height


def test_dinamica_recupera_un_adelanto_y_el_neto_es_la_suma(generar_tablas):
    verdadero = math.log(0.97)
    ayer = pl.col(PROMO).shift(1).over(SERIE).fill_null(0)
    p = _panel(generar_tablas, efecto=ayer * verdadero, ruido=0.05)
    d = efectos.dinamica(efectos.muestra(efectos.preparar(p)))
    t1 = d["coeficientes"]["t-1"]
    assert t1["ic95"][0] < verdadero < t1["ic95"][1]
    assert d["coeficientes"]["t"]["ic95"][0] < 0 < d["coeficientes"]["t"]["ic95"][1]
    betas = [c["beta"] for c in d["coeficientes"].values()]
    assert d["neto"]["beta"] == pytest.approx(sum(betas), abs=1e-12)
    assert d["suma_rezagos"]["beta"] == pytest.approx(sum(betas[1:]), abs=1e-12)
    assert d["adelanto"]
    # Sin adelanto no se declara
    sin = efectos.dinamica(efectos.muestra(efectos.preparar(_panel(generar_tablas, ruido=0.05))))
    assert not sin["adelanto"]


def test_efectos_cruzados_recuperan_un_arrastre(generar_tablas):
    otras = pl.col(PROMO).sum().over("store_id", "date") - pl.col(PROMO)
    p = _panel(generar_tablas, efecto=0.02 * otras, ruido=0.05)
    c = efectos.cruzado(efectos.muestra(efectos.preparar(p)))
    assert c["otras_categorias"]["ic95"][0] < 0.02 < c["otras_categorias"]["ic95"][1]
    assert c["interpretacion"] == efectos.ARRASTRE
    canibal = efectos.cruzado(efectos.muestra(efectos.preparar(_panel(generar_tablas, efecto=-0.02 * otras))))
    assert canibal["interpretacion"] == efectos.CANIBALIZACION


def test_p_por_aleatorizacion_a_mano_y_su_minimo(generar_tablas):
    # #{|β_b| ≥ 0.2} = 2 (−0.3 y 0.2): p = (1 + 2) / (3 + 1)
    assert potencia.p_fisher(0.2, np.array([0.1, -0.3, 0.2])) == pytest.approx(0.75)
    assert potencia.p_fisher(-0.5, np.array([0.1, -0.3, 0.2])) == pytest.approx(0.25)
    # Con un efecto fuerte ninguna permutación lo alcanza: el p es el mínimo, 1 / (B + 1)
    p = _panel(generar_tablas, efecto=pl.col(PROMO) * 0.3)
    m = efectos.muestra(efectos.preparar(p))
    principal, _ = efectos.principal(m)
    dis = potencia.diseno(m, Y, PROMO)
    r = potencia.aleatorizacion(dis, principal["beta"], Parametros(permutaciones=50))
    assert r["superan"] == 0 and r["p_valor"] == pytest.approx(1 / 51) == r["p_minimo"]


def test_la_permutacion_conserva_las_promociones_de_cada_fecha():
    rng = np.random.default_rng(0)
    fecha = np.repeat(np.arange(5), [3, 7, 1, 6, 4])
    fecha = fecha[rng.permutation(fecha.size)]  # las filas de una fecha no van juntas
    d = (rng.random(fecha.size) < 0.4).astype(float)
    permutar = potencia.permutador(d, fecha)
    sorteos = [permutar(np.random.default_rng(1)) for _ in range(2)]
    np.testing.assert_array_equal(sorteos[0], sorteos[1])  # misma semilla, misma permutación
    distintas = 0
    otro = np.random.default_rng(2)
    for _ in range(20):
        x = permutar(otro)
        for f in range(5):
            assert sorted(x[fecha == f]) == sorted(d[fecha == f])
        distintas += not np.array_equal(x, d)
    assert distintas > 0
    with pytest.raises(ValueError, match="nulos"):
        potencia.permutador(np.array([1.0, np.nan]), np.array([0, 0]))


def test_calibracion_aa_y_potencia(generar_tablas):
    inf, sup = potencia.intervalo_aa(500, 0.05, 0.99)
    assert (inf * 500, sup * 500) == pytest.approx(stats.binom.interval(0.99, 500, 0.05))
    assert inf < 0.05 < sup
    factor = float(stats.t.ppf(0.975, 79) + stats.t.ppf(0.8, 79))
    assert potencia.mde(0.01, 79)["mde_log"] == pytest.approx(factor * 0.01)
    assert potencia.mde(0.01, 79)["mde_pct"] == pytest.approx(100 * math.expm1(factor * 0.01))

    m = efectos.muestra(efectos.preparar(_panel(generar_tablas)))
    aa = potencia.calibracion_aa(potencia.diseno(m, Y, PROMO), Parametros(placebos_aa=60))
    assert aa["gl"] == 19 and aa["placebos"] == 60
    assert aa["inferencia_principal"] == ("agrupada_por_tienda" if aa["dentro"] else "aleatorizacion")


def test_el_panel_semanal_solo_lleva_semanas_completas(generar_tablas):
    m = efectos.muestra(efectos.preparar(_panel(generar_tablas, dias=30)))
    s = potencia.panel_semanal(m)
    # 2023-01-01 es domingo: su semana tiene un día. Del 01-02 al 01-29 hay 4 semanas; la del 01-30, 1 día
    assert s["semana"].unique().sort().to_list() == [INICIO + timedelta(days=1 + 7 * k) for k in range(4)]
    assert (s["dias"] == 7).all()
    # Hay nulos de units_sold, así que a alguna serie le faltan semanas
    assert m.height < 20 * 6 * 30 and s.height < 20 * 6 * 4
    p = Parametros(placebos_semanal=20, semanas_rejilla=(2, 3, 4, 6), placebos_por_ventana=10)
    semanal = potencia.diseno_semanal(m, p)
    assert semanal["semanas"] == 4 and semanal["desde"] == "2023-01-02" and semanal["hasta"] == "2023-01-29"
    assert [f["efecto_pct"] for f in semanal["semanas_necesarias"]] == pytest.approx([2, 5, 10])
    # Ventanas no traslapadas de las 4 semanas: dos de 2, una de 3 (sobra una semana), una de 4 y ninguna de 6
    assert [(f["semanas"], f["ventanas"]) for f in semanal["por_semanas"]] == [(2, 2), (3, 1), (4, 1)]
    lunes = [INICIO + timedelta(days=7 * k) for k in range(5)]
    assert potencia.ventanas(lunes, 2) == [lunes[0:2], lunes[2:4]]
    for f in semanal["por_semanas"]:
        assert f["mde_log"] == pytest.approx(potencia.factor_mde(f["gl"]) * f["ee"])
    assert "supuesto" not in semanal


def test_causal_de_punta_a_punta_es_reproducible(generar_tablas, tmp_path):
    ruta = tmp_path / "panel.parquet"
    _panel(generar_tablas, n_tiendas=40).write_parquet(ruta)
    r = etapa.ejecutar(ruta, tmp_path / "a.json", RAPIDOS)
    etapa.ejecutar(ruta, tmp_path / "b.json", RAPIDOS)
    texto = (tmp_path / "a.json").read_text()
    assert texto == (tmp_path / "b.json").read_text()
    assert json.loads(texto) == r
    assert list(r) == [
        "fuente",
        "muestra",
        "balance",
        "efecto_principal",
        "canales",
        "robustez",
        "calibracion_aa",
        "heterogeneidad",
        "dinamica",
        "cruzado",
        "potencia",
        "negocio",
        "supuestos",
        "definiciones",
        "conclusion",
    ]
    assert "STR_" not in texto  # sin filas ni series
    c = r["conclusion"]
    assert c["aleatorizacion_se_sostiene"] == r["balance"]["se_sostiene"]
    assert c["inferencia_principal"] == r["calibracion_aa"]["inferencia_principal"]
    e = r["efecto_principal"]
    assert e["p_aleatorizacion"] == r["robustez"]["aleatorizacion"]["p_valor"]
    assert e["n"] == r["muestra"]["filas"] and e["gl"] == 39
    assert set(r["canales"]) == {*efectos.RESULTADOS, "transacciones_efectivo_tarjeta"}
    # El precio por unidad es monto − unidades en log: sus β cuadran
    canales = r["canales"]
    assert canales["precio_unitario"]["beta"] == pytest.approx(
        canales["monto"]["beta"] - canales["unidades"]["beta"], abs=1e-5
    )
    assert set(r["negocio"]["por_categoria"]) == set(CATEGORIAS)
    assert r["supuestos"]["parametros"]["permutaciones"] == 40


def test_sin_panel_causal_pide_correr_calidad(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "get_config", lambda: Config(_env_file=None, data_dir=tmp_path, artifacts_dir=tmp_path)
    )
    with pytest.raises(SystemExit):
        cli.main(["causal"])
    assert "corre antes `demanda calidad`" in capsys.readouterr().err


HUECOS = (
    (pl.col("store_id") == "STR_002") & pl.col("date").is_between(date(2023, 1, 20), date(2023, 1, 22))
) | (
    (pl.col("store_id") == "STR_005") & (pl.col("category") == "Ropa") & (pl.col("date") == date(2023, 2, 10))
)


def _chi2_pearson(tabla: np.ndarray) -> float:
    """La χ² de una tabla 2 × 2 por su fórmula cerrada, sin corrección de continuidad."""
    (a, b), (c, d) = np.asarray(tabla, dtype=float)
    return (a + b + c + d) * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))


def test_holm_es_sobre_toda_la_familia(generar_tablas):
    def por_formato(p: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
        tasa = np.where(p["store_format"].to_numpy() == "Bodega", 0.15, 0.25)
        return (rng.random(p.height) < tasa).astype(int)

    pruebas = _evaluar_balance(_panel(generar_tablas, por_formato, n_tiendas=40))["familia"]["pruebas"]
    p_valores = [f["p_valor"] for f in pruebas]
    esperado = multipletests(p_valores, method="holm")[1]
    assert [f["p_holm"] for f in pruebas] == pytest.approx(list(esperado), rel=1e-12)
    # Con varios p chicos, Holm no es Bonferroni: la prueba distingue las dos
    assert not np.allclose(esperado, np.minimum(1, len(p_valores) * np.array(p_valores)))

    # Heterogeneidad: una sola familia con las 6 categorías y los 3 formatos
    en_bebidas = (pl.col(PROMO) == 1) & (pl.col("category") == "Bebidas")
    p = _panel(generar_tablas, efecto=pl.when(en_bebidas).then(math.log(1.05)).otherwise(0.0), dias=112)
    het = efectos.heterogeneidad(efectos.muestra(efectos.preparar(p)))
    grupos = [het[d]["grupos"][g] for d in efectos.HETEROGENEIDAD for g in het[d]["grupos"]]
    esperado = multipletests([g["p_valor"] for g in grupos], method="holm")[1]
    assert [g["p_holm"] for g in grupos] == pytest.approx(list(esperado), rel=1e-12)
    # Bebidas tiene el menor p: con 9 pruebas su p de Holm es 9 · p, no el 6 · p de su dimensión
    bebidas = het["categoria"]["grupos"]["Bebidas"]
    assert bebidas["p_holm"] == pytest.approx(9 * bebidas["p_valor"])


def test_b3_con_efectos_fijos_no_confunde_el_nivel_de_la_serie(generar_tablas):
    # Las series que venden más tienen más promoción, pero la promoción no sigue a las ventas recientes
    def por_nivel(p: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
        rango = p.select(pl.col("alfa").rank("dense") / pl.col("alfa").n_unique()).to_series().to_numpy()
        return (rng.random(p.height) < 0.05 + 0.3 * rango).astype(int)

    m = efectos.muestra(efectos.preparar(_panel(generar_tablas, por_nivel, n_tiendas=40)))
    adelantos = [adelanto(k) for k in REZAGOS]
    assert balance.ventas_previas(m, adelantos, Y)["cumple"]
    # Sin efectos fijos, el nivel de la serie sí parecería selección por ventas previas
    df = m.filter(pl.all_horizontal(pl.col(a).is_not_null() for a in adelantos))
    assert wald(estimar(df, Y, adelantos, efectos=((),)), np.eye(len(adelantos))).p_valor < 1e-6


def test_b4_y_b5_contra_tablas_a_mano(generar_tablas):
    p = _panel(generar_tablas, quitar=HUECOS)
    preparado = efectos.preparar(p)
    existentes = p.filter(~pl.col("flag_fila_faltante"))
    con_efectivo = existentes.filter(pl.col("cash_transactions").is_not_null())

    def tabla(df: pl.DataFrame, evento: pl.Expr) -> np.ndarray:
        marca = df.select(evento).to_series().to_numpy()
        promo = df[PROMO].to_numpy()
        return np.array([[np.sum((promo == i) & (marca == j)) for j in (0, 1)] for i in (0, 1)])

    b5 = {f["prueba"]: f for f in balance.atricion(preparado)}
    for prueba, df, evento in (
        ("B5:units_sold_nulo", existentes, pl.col("units_sold").is_null()),
        ("B5:conteo_inconsistente", con_efectivo, pl.col("flag_conteo_inconsistente")),
    ):
        t = tabla(df, evento)
        assert b5[prueba]["n"] == df.height == t.sum()
        assert b5[prueba]["chi2"] == pytest.approx(_chi2_pearson(t), rel=1e-10)
    # La bandera solo se evalúa donde hay efectivo
    assert con_efectivo.height < existentes.height

    # B4: los pares (t − k, t) de cada serie en la malla, con las dos promociones observadas
    for k in (1, 7):
        t = np.zeros((2, 2))
        for (_, _), serie in p.group_by(SERIE):
            x = serie.sort("date")[PROMO].to_list()
            for i in range(k, len(x)):
                if x[i] is not None and x[i - k] is not None:
                    t[x[i - k], x[i]] += 1
        r = balance.independencia_serial(preparado, k, rezago(k))
        assert r["n"] == t.sum()
        assert r["chi2"] == pytest.approx(_chi2_pearson(t), rel=1e-10)


def test_b4_detecta_dependencia_a_siete_dias_de_punta_a_punta(generar_tablas):
    def semanal(p: pl.DataFrame, rng: np.random.Generator) -> np.ndarray:
        # Cada serie repite su promoción de hace una semana con probabilidad 0.8; si no, sortea al 20 %
        dias = p["date"].n_unique()
        x = (rng.random(p.height) < 0.2).astype(int)
        repetir = rng.random(p.height) < 0.8
        for i in range(p.height):  # el panel va ordenado por serie y fecha, sin huecos
            if i % dias >= 7 and repetir[i]:
                x[i] = x[i - 7]
        return x

    r = etapa.analizar(_panel(generar_tablas, semanal, n_tiendas=40), RAPIDOS)
    b4 = {f["prueba"]: f for f in r["balance"]["familia"]["pruebas"] if f["prueba"].startswith("B4")}
    assert b4["B4:t-7"]["p_holm"] < 1e-6 and b4["B4:t-7"]["tasa_si_hubo_pct"] > 70
    assert b4["B4:t-1"]["p_holm"] >= 0.05
    assert "B4:t-7" in r["balance"]["fallan"] and "B4:t-1" not in r["balance"]["fallan"]


def test_dinamica_usa_los_rezagos_de_la_malla(generar_tablas):
    p = _panel(generar_tablas, quitar=HUECOS)
    m = efectos.muestra(efectos.preparar(p))
    # Una fila entra si existen las 7 filas anteriores de su serie en el calendario, aunque no tengan
    # units_sold: la promoción se observa igual
    existen = set(p.filter(~pl.col("flag_fila_faltante")).select(*SERIE, "date").iter_rows())
    esperado = sum(
        all((s, c, f - timedelta(days=k)) in existen for k in REZAGOS)
        for s, c, f in m.select(*SERIE, "date").iter_rows()
    )
    assert efectos.dinamica(m)["n"] == esperado < m.height - 7 * 120


def _estimacion(valor: float, inf: float, sup: float) -> Estimacion:
    return Estimacion(valor, 0.01, 1.0, 79, 0.3, inf, sup)


def test_adelanto_y_arrastre_se_deciden_por_el_ic(generar_tablas):
    # Casos nulos con el punto de un lado y el IC cruzando el cero: ni adelanto ni efecto cruzado
    d = efectos.dinamica(efectos.muestra(efectos.preparar(_panel(generar_tablas, semilla=4))))
    assert d["suma_rezagos"]["beta"] < 0 < d["suma_rezagos"]["ic95"][1]
    assert not d["adelanto"]
    for semilla, signo in ((10, 1), (0, -1)):
        c = efectos.cruzado(efectos.muestra(efectos.preparar(_panel(generar_tablas, semilla=semilla))))
        otras = c["otras_categorias"]
        assert signo * otras["beta"] > 0 and otras["ic95"][0] < 0 < otras["ic95"][1]
        assert c["interpretacion"] == efectos.SIN_EFECTO
    # La regla con estimaciones hechas a mano
    assert not efectos.hay_adelanto(_estimacion(-0.01, -0.03, 0.01))
    assert efectos.hay_adelanto(_estimacion(-0.02, -0.04, -0.001))
    assert efectos.interpretar_cruzado(_estimacion(0.01, -0.01, 0.03)) == efectos.SIN_EFECTO
    assert efectos.interpretar_cruzado(_estimacion(0.02, 0.001, 0.04)) == efectos.ARRASTRE
    assert efectos.interpretar_cruzado(_estimacion(-0.02, -0.04, -0.001)) == efectos.CANIBALIZACION


def test_robustez_agrupada_doble_y_ponderada_a_mano(generar_tablas):
    m = efectos.muestra(efectos.preparar(_panel(generar_tablas, n_tiendas=8, dias=42)))
    rob = efectos.robustez(m)
    # Cameron, Gelbach y Miller: V = V_tienda + V_fecha − V_tienda×fecha, cada una con su factor CRV1 (K = 1)
    factores = [codigos(m, SERIE), codigos(m, ("date",))]
    datos = absorber(m.select(Y, PROMO).to_numpy(), factores).datos
    y, x = datos[:, 0], datos[:, 1]
    u = y - (x @ y) / (x @ x) * x

    def varianza(grupos: np.ndarray) -> float:
        g = densos(grupos)
        n_grupos = g.max() + 1
        s = np.bincount(g, weights=x * u)
        return n_grupos / (n_grupos - 1) * (s @ s) / (x @ x) ** 2

    cgm = varianza(codigos(m, TIENDA)) + varianza(factores[1]) - varianza(codigos(m, ("store_id", "date")))
    doble = rob["agrupado_tienda_y_fecha"]
    assert doble["ee"] == pytest.approx(math.sqrt(cgm), rel=1e-9)
    assert doble["gl"] == 7  # min(8 tiendas, 42 fechas) − 1
    assert doble["ee"] != pytest.approx(efectos.principal(m)[0]["ee"], rel=1e-3)

    # Ponderada: WLS con indicadoras de serie y de fecha, con la mediana de la serie como peso
    w = m.select(pl.col("units_sold").median().over(SERIE)).to_series().to_numpy()
    serie, fecha = factores
    indicadoras = np.column_stack(
        [serie[:, None] == np.arange(serie.max() + 1), fecha[:, None] == np.arange(1, fecha.max() + 1)]
    ).astype(float)
    wls = sm.WLS(m[Y].to_numpy(), np.column_stack([m[PROMO].to_numpy(), indicadoras]), weights=w).fit()
    ponderada = rob["ponderado_por_volumen"]["beta"]
    assert ponderada == pytest.approx(wls.params[0], abs=1e-9)
    assert ponderada != pytest.approx(efectos.principal(m)[0]["beta"], abs=1e-6)


def test_los_canales_son_combinaciones_de_logs(generar_tablas):
    c = efectos.canales(efectos.muestra(efectos.preparar(_panel(generar_tablas))))
    assert abs(c["canasta"]["beta"]) > 1e-4
    assert c["canasta"]["beta"] == pytest.approx(
        c["unidades"]["beta"] - c["transacciones"]["beta"], abs=1e-12
    )
    assert c["precio_unitario"]["beta"] == pytest.approx(
        c["monto"]["beta"] - c["unidades"]["beta"], abs=1e-12
    )
    assert [n for n, v in c.items() if not v["secundario"]] == ["unidades"]


def test_la_permutacion_identidad_reproduce_el_ajuste(generar_tablas):
    m = efectos.muestra(efectos.preparar(_panel(generar_tablas)))
    principal, _ = efectos.principal(m)
    dis = potencia.diseno(m, Y, PROMO)
    betas, ts = potencia.placebos(dis.y, dis.factores, dis.tienda, lambda: dis.d, 3, lote=2)
    assert betas == pytest.approx([principal["beta"]] * 3, abs=1e-12)
    assert ts == pytest.approx([principal["t"]] * 3, rel=1e-9)


def test_diseno_semanal_es_estimar_con_la_misma_asignacion(generar_tablas):
    m = efectos.muestra(efectos.preparar(_panel(generar_tablas)))
    p = Parametros(placebos_semanal=2, semanas_rejilla=())
    r = potencia.diseno_semanal(m, p)
    s = potencia.panel_semanal(m)
    rng = np.random.default_rng(p.semilla_semanal)
    ajustes = []
    for _ in range(2):
        d = s.with_columns(pl.Series("d", (rng.random(s.height) < 0.5).astype(float)))
        ajustes.append(estimar(d, "log_unidades", ["d"], efectos=(SERIE, ("semana",))))
    estimaciones = [coeficiente(a, "d") for a in ajustes]
    assert r["ee"] == pytest.approx(np.mean([e.ee for e in estimaciones]), rel=1e-9)
    assert r["de_beta_placebo"] == pytest.approx(np.std([e.valor for e in estimaciones], ddof=1), rel=1e-6)
    assert ajustes[0].gl == 19
    assert r["mde_actual"]["mde_log"] == pytest.approx(potencia.factor_mde(19) * r["ee"], rel=1e-12)
    assert r["por_semanas"] == [] and all(f["semanas"] is None for f in r["semanas_necesarias"])


def test_cada_ventana_usa_solo_sus_semanas(generar_tablas):
    lunes = [date(2023, 1, 2) + timedelta(weeks=k) for k in range(5)]
    assert potencia.ventanas(lunes, 2) == [lunes[0:2], lunes[2:4]]
    m = efectos.muestra(efectos.preparar(_panel(generar_tablas)))
    tabla = potencia.por_semanas(
        potencia.panel_semanal(m), np.random.default_rng(0), Parametros(semanas_rejilla=(2, 4))
    )
    dos, cuatro = tabla
    assert (dos["semanas"], cuatro["semanas"]) == (2, 4) and dos["ventanas"] > cuatro["ventanas"]
    # Con la mitad de semanas por ventana el EE sube cerca de √2; con el panel completo no cambiaría
    assert dos["ee"] > 1.2 * cuatro["ee"]


def test_semanas_necesarias_compara_el_mde_en_log():
    tabla = [{"semanas": 4, "mde_log": 0.03}, {"semanas": 2, "mde_log": 0.0495}]
    # log(1.05) = 0.0488 < 0.0495 ≤ 0.05: con 2 semanas el MDE es 5.07 % y no alcanza el 5 %
    assert potencia.semanas_necesarias(tabla, 0.05) == 4
    assert potencia.semanas_necesarias(tabla, 0.06) == 2
    assert potencia.semanas_necesarias(tabla, 0.02) is None


def test_p_principal_sigue_a_la_inferencia_principal(generar_tablas):
    p = _panel(generar_tablas, n_tiendas=40)
    e = etapa.analizar(p, RAPIDOS)["efecto_principal"]
    assert e["inferencia_principal"] == "agrupada_por_tienda"
    assert e["p_valor_principal"] == e["p_valor"] != e["p_aleatorizacion"]
    # Con un intervalo A/A de ancho cero (solo vale exactamente 5 %), la A/A sale y manda la aleatorización
    r = etapa.analizar(p, replace(RAPIDOS, nivel_aa=1e-9))
    assert not r["calibracion_aa"]["dentro"]
    e = r["efecto_principal"]
    assert e["inferencia_principal"] == "aleatorizacion"
    assert e["p_valor_principal"] == e["p_aleatorizacion"] != e["p_valor"]
    assert r["conclusion"]["p_valor_principal"] == e["p_aleatorizacion"]
