"""Datos sintéticos con la misma forma que los reales. Los tests nunca leen `data/raw/`."""

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import polars as pl
import pytest

from demanda.backtest.protocolo import CONOCIDAS, DE_LA_SENAL, Protocolo, ultima_fecha
from demanda.datos import panel
from demanda.datos.contratos import CATEGORIAS, FORMATOS, NSE, REGIONES, SERIE, Tablas
from demanda.modelos import chronos2

# La config de todo el suite (y de los subprocesos que lance) lee este `.env`, que no existe: ninguna
# prueba ve las keys reales
SIN_ENV = Path(__file__).parent / "sin.env"
os.environ["DEMANDA_ENV_FILE"] = str(SIN_ENV)

INICIO = date(2023, 1, 1)


def generar(
    n_tiendas: int = 3,
    categorias: tuple[str, ...] = ("Abarrotes", "Bebidas"),
    dias: int = 56,
    ventana_senal: tuple[int, int] = (-5, 1),
    semilla: int = 0,
) -> Tablas:
    """Tablas válidas con los problemas conocidos de los datos reales, en proporciones parecidas.

    - Cada quinta fila tiene conteos inconsistentes (`cash + card < total`) y `avg_ticket` usa `cash + card`.
    - ~6 % de nulos simultáneos en `cash_transactions` y `amount_cash`; `avg_ticket` es nulo en parte de esos.
    - ~3 % de nulos en `units_sold` y la señal nula en las dos primeras fechas.
    - `replenishment_signal` = 0.53 × la media de `units_sold` de t + `desde` a t + `hasta` (`ventana_senal`).
    """
    rng = np.random.default_rng(semilla)
    fechas = pl.date_range(INICIO, INICIO + timedelta(days=dias - 1), "1d", eager=True)
    calendario = pl.DataFrame({"date": fechas}).with_columns(
        (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("day_of_week"),
        pl.col("date").dt.strftime("%A").alias("day_name"),
        pl.col("date").dt.week().cast(pl.Int64).alias("week_of_year"),
        pl.col("date").dt.month().cast(pl.Int64).alias("month"),
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
        pl.col("date").dt.quarter().cast(pl.Int64).alias("quarter"),
        pl.lit("Invierno").alias("season"),
        (pl.col("date") == INICIO).alias("is_holiday"),
        pl.when(pl.col("date") == INICIO).then(pl.lit("Año Nuevo")).alias("holiday_name"),
        (pl.col("date").dt.day() == 15).alias("is_payday"),
        (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
        pl.lit(False).alias("is_navidad_season"),
        pl.lit(False).alias("is_buen_fin"),
        pl.lit(False).alias("is_semana_santa"),
    )

    tiendas = pl.DataFrame(
        {
            "store_id": [f"STR_{i:03d}" for i in range(1, n_tiendas + 1)],
            "store_format": [FORMATOS[i % len(FORMATOS)] for i in range(n_tiendas)],
            "region": [REGIONES[i % len(REGIONES)] for i in range(n_tiendas)],
            "size_sqm": [5000 + 1000 * i for i in range(n_tiendas)],
            "num_checkouts": [10 + i for i in range(n_tiendas)],
            "opening_year": [2015] * n_tiendas,
            "socioeconomic_level": [NSE[i % len(NSE)] for i in range(n_tiendas)],
            "has_pharmacy": [i % 2 == 0 for i in range(n_tiendas)],
            "has_fuel_station": [False] * n_tiendas,
        }
    )

    # La señal puede usar días posteriores: se generan ventas de más y se recortan al final
    desde, hasta = ventana_senal
    extra = pl.date_range(INICIO, INICIO + timedelta(days=dias - 1 + max(hasta, 0)), "1d", eager=True)
    malla = (
        pl.DataFrame({"date": extra})
        .join(tiendas.select("store_id"), how="cross")
        .join(pl.DataFrame({"category": list(categorias)}), how="cross")
        .sort([*SERIE, "date"])
    )
    n = malla.height
    nivel = malla.select(
        pl.col("store_id").rank("dense") * 100 + pl.col("category").rank("dense") * 40
    ).to_series()
    fin_de_semana = malla.select(pl.col("date").dt.weekday() >= 6).to_series()
    unidades = np.round(nivel.to_numpy() * np.where(fin_de_semana, 1.5, 1.0) * rng.lognormal(0, 0.2, n))
    total = np.maximum(np.round(unidades / 2.5), 2).astype(np.int64)
    tarjeta = np.round(total * 0.6).astype(np.int64)
    efectivo = (total - tarjeta).astype(np.float64)
    efectivo[np.arange(n) % 5 == 0] -= 1  # conteos inconsistentes
    monto = np.round(total * 300 * rng.lognormal(0, 0.1, n), 2)
    monto_efectivo = np.round(monto * efectivo / (efectivo + tarjeta), 2)
    ticket = np.round(monto / (efectivo + tarjeta), 2)

    senal = malla.with_columns(pl.Series("u", unidades)).select(
        (pl.col("u").rolling_mean(hasta - desde + 1, min_samples=1).shift(-hasta).over(SERIE) * 0.53).round(2)
    )

    sin_efectivo = rng.random(n) < 0.06
    sin_ticket = sin_efectivo & (rng.random(n) < 0.4)
    sin_unidades = rng.random(n) < 0.03
    transacciones = malla.with_columns(
        pl.Series("total_transactions", total),
        pl.Series("cash_transactions", np.where(sin_efectivo, np.nan, efectivo)).fill_nan(None),
        pl.Series("card_transactions", tarjeta),
        pl.Series("amount_total", monto),
        pl.Series("amount_cash", np.where(sin_efectivo, np.nan, monto_efectivo)).fill_nan(None),
        pl.Series("amount_card", np.round(monto - monto_efectivo, 2)),
        pl.Series("units_sold", np.where(sin_unidades, np.nan, unidades)).fill_nan(None),
        pl.Series("avg_ticket", np.where(sin_ticket, np.nan, ticket)).fill_nan(None),
        pl.Series("has_promotion", (rng.random(n) < 0.2).astype(np.int64)),
        senal.to_series().alias("replenishment_signal"),
    ).with_columns(pl.when(pl.col("date") >= INICIO + timedelta(days=2)).then(pl.col("replenishment_signal")))
    return Tablas(transacciones.filter(pl.col("date").is_in(fechas.implode())), tiendas, calendario)


def escribir_csv(tablas: Tablas, carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    tablas.transacciones.write_csv(carpeta / "transactions.csv")
    tablas.tiendas.write_csv(carpeta / "stores.csv")
    tablas.calendario.write_csv(carpeta / "calendar.csv")
    return carpeta


@pytest.fixture
def tablas() -> Tablas:
    return generar()


@pytest.fixture
def generar_tablas():
    """La función `generar`, para los tests que necesitan otros parámetros."""
    return generar


@pytest.fixture
def carpeta_csv(tmp_path: Path):
    """Escribe unas tablas como CSV en una carpeta temporal y devuelve la carpeta."""
    return lambda tablas: escribir_csv(tablas, tmp_path / "raw")


def alterar_futuro(p: pl.DataFrame, origen: date) -> pl.DataFrame:
    """Cambia, después de `origen`, todas las columnas que no se conocen de antemano, y también la señal y su
    bandera fechadas en `origen`. Un pronóstico o una feature sin fugas no debe moverse."""
    despues = pl.col("date") > origen

    def alterar(columna: str) -> pl.Expr:
        nuevo = ~pl.col(columna) if p.schema[columna] == pl.Boolean else pl.col(columna).fill_null(0) * 3 + 7
        cuando = despues | (pl.col("date") == origen) if columna in DE_LA_SENAL else despues
        return pl.when(cuando).then(nuevo).otherwise(pl.col(columna)).alias(columna)

    return p.with_columns(alterar(c) for c in p.columns if c not in CONOCIDAS)


@pytest.fixture(scope="session")
def panel_sintetico() -> pl.DataFrame:
    """Panel de 70 días: alcanza para dos cortes semanales con ocho semanas de historia."""
    return panel.construir(generar(dias=70))


@pytest.fixture(scope="session")
def alterador():
    """La función `alterar_futuro`, para las pruebas de fugas."""
    return alterar_futuro


class PipelineDoble:
    """Doble de `Chronos2Pipeline.predict_df` que calcula su salida con todo lo que recibe: cada valor del
    contexto (objetivo y covariables, pesado por su posición) y las covariables de cada día futuro. Si al
    modelo le llegara algo del futuro, el pronóstico cambiaría. Guarda los argumentos de cada llamada."""

    def __init__(self) -> None:
        self.llamadas: list[dict] = []
        self.cerrado = False

    def cerrar(self) -> None:
        self.cerrado = True

    def predict_df(self, df, future_df, **kwargs):
        self.llamadas.append({"df": df, "future_df": future_df, **kwargs})
        h, niveles = kwargs["prediction_length"], kwargs["quantile_levels"]
        filas = []
        for item, contexto in df.groupby("item_id", sort=True):
            contexto = contexto.sort_values("timestamp")
            valores = contexto.drop(columns=["item_id", "timestamp"]).to_numpy(dtype=float)
            pesos = np.arange(1, len(contexto) + 1)[:, None]
            huella = np.nansum(valores * pesos) / pesos.sum()
            nivel = np.nanmean(contexto["target"].to_numpy(dtype=float))
            conocido = future_df[future_df["item_id"] == item].sort_values("timestamp")
            covariables = conocido.drop(columns=["item_id", "timestamp"]).to_numpy(dtype=float)
            fechas = pd.date_range(contexto["timestamp"].max(), periods=h + 1, freq="D")[1:]
            for j in range(h):
                mediana = nivel * (
                    1 + 0.05 * np.nansum(covariables[j] * np.arange(1, covariables.shape[1] + 1))
                )
                mediana += 1e-3 * huella
                fila = {"item_id": item, "timestamp": fechas[j]}
                fila |= {"target_name": "target", "predictions": mediana}
                filas.append(fila | {str(q): mediana * (0.5 + q) for q in niveles})
        return pd.DataFrame(filas)


@pytest.fixture(autouse=True)
def chronos_sin_descargas(monkeypatch):
    """En las pruebas, el `chronos2` del registro usa el doble: no descarga ni carga pesos."""
    monkeypatch.setattr(chronos2, "CARGADOR", PipelineDoble)


@pytest.fixture
def doble_chronos():
    """La clase del doble, para las pruebas que revisan cómo se llama al pipeline."""
    return PipelineDoble


@pytest.fixture(scope="session")
def panel_monitor() -> pl.DataFrame:
    """Panel de 20 tiendas × 6 categorías × 70 días para el monitor: cada región tiene 4 tiendas y cada
    segmento, cientos de celdas por semana."""
    return panel.construir(generar(n_tiendas=20, categorias=CATEGORIAS, dias=70))


class ArtefactosE5(NamedTuple):
    panel: Path
    reporte: Path
    pronosticos: Path
    metricas: Path
    seleccion: Path
    calibracion: Path


@pytest.fixture(scope="session")
def artefactos_e5(tmp_path_factory) -> ArtefactosE5:
    """El pipeline sintético hasta la calibración, con lo que leen `decision` y `drift`: un panel de 112 días
    (del 2023-01-01 al 2023-04-22), un backtest de 4 cortes con el sistema actual y la MA7, su selección y la
    calibración en los 2 orígenes previos (2023-03-11 y 03-18).

    Los cortes tienen los cuatro eventos: 03-25 es `resto`, 04-01 tiene un Buen Fin el 04-03, 04-08 es la
    semana posterior y 04-15 tiene un día de temporada navideña el 04-20."""
    from demanda.backtest import bucle, calibracion, segmentos
    from demanda.modelos import Contexto, crear
    from demanda.seleccion import campeon

    carpeta = tmp_path_factory.mktemp("e5")
    rutas = ArtefactosE5(
        carpeta / "panel.parquet",
        carpeta / "reporte.json",
        carpeta / "pronosticos.parquet",
        carpeta / "metricas.json",
        carpeta / "seleccion.json",
        carpeta / "calibracion.parquet",
    )
    p = panel.construir(generar(n_tiendas=10, categorias=CATEGORIAS, dias=112, semilla=1)).with_columns(
        (pl.col("date") == date(2023, 4, 3)).alias("is_buen_fin"),
        (pl.col("date") == date(2023, 4, 20)).alias("is_navidad_season"),
    )
    p.write_parquet(rutas.panel)
    rutas.reporte.write_text(json.dumps({"fuga_replenishment_signal": {"factor": 0.53}}))

    protocolo = Protocolo(n_cortes=4)
    modelos = crear(Contexto(factor_senal=0.53), ["sistema_actual", "ma7"])
    resultado = bucle.correr(p, modelos, protocolo)
    resultado.pronosticos.write_parquet(rutas.pronosticos)
    metricas = {
        "protocolo": {
            "horizonte": protocolo.horizonte,
            "origenes": [str(o) for o in protocolo.origenes(ultima_fecha(p))],
            "modelos": list(modelos),
            "evento_por_corte": segmentos.evento_por_corte(resultado, p, protocolo.horizonte),
        },
        "segmentos": segmentos.tabla(resultado, p, protocolo, list(modelos)).to_dicts(),
    }
    rutas.metricas.write_text(json.dumps(metricas))
    campeon.ejecutar(rutas.metricas, rutas.pronosticos, rutas.seleccion)
    calibracion.ejecutar(
        rutas.panel, rutas.reporte, rutas.seleccion, rutas.calibracion, protocolo=protocolo, n_origenes=2
    )
    return rutas


# --- E7: artefactos sintéticos para el servidor MCP y los agentes ---------------------------------------

MODELOS_E7 = ("sistema_actual", "lightgbm")
ORIGENES_E7 = ("2023-11-02", "2023-11-09")


def _metricas_e7() -> dict:
    filas = []
    valores = {
        ("global", "global"): {
            "sistema_actual": (27.2929, 1.6794, 30.1, None),
            "lightgbm": (20.3353, -3.5995, 24.2, 68.7),
        },
        ("evento", "resto"): {
            "sistema_actual": (10.9786, 0.8, 15.2, None),
            "lightgbm": (11.3004, -1.2, 15.9, 70.1),
        },
        ("evento", "buen_fin"): {
            "sistema_actual": (49.8852, -49.8852, 52.3, None),
            "lightgbm": (48.8892, -48.1, 51.0, 40.2),
        },
    }
    for (segmento, nivel), por_modelo in valores.items():
        for modelo, (wape, sesgo, wape_d, cob) in por_modelo.items():
            base = {"segmento": segmento, "nivel": nivel, "modelo": modelo, "n": 90, "n_total": 112}
            filas += [
                {**base, "granularidad": "semanal", "metrica": "wape_pct", "valor": wape},
                {**base, "granularidad": "semanal", "metrica": "sesgo_pct", "valor": sesgo},
                {**base, "granularidad": "diaria", "metrica": "wape_pct", "valor": wape_d},
            ]
            if cob is not None:
                filas.append({**base, "granularidad": "diaria", "metrica": "cobertura_80_pct", "valor": cob})
    return {
        "protocolo": {"modelos": list(MODELOS_E7), "origenes": list(ORIGENES_E7), "horizonte": 7},
        "cobertura": {"semanal": {"pares_total": 112, "pares_completos": 90, "pct": 80.36}},
        "segmentos": filas,
        "definiciones": {
            "metricas": {
                "wape_pct": "100 · Σ|y − ŷ| / Σy",
                "sesgo_pct": "100 · Σ(ŷ − y) / Σy",
                "cobertura_80_pct": "% en [p10, p90]",
            },
            "granularidad": {"semanal": "suma de T+1 a T+h"},
            "segmentos": {"evento": "buen_fin, navidad, post_buen_fin o resto"},
        },
    }


def _seleccion_e7() -> dict:
    criterios = {
        "gana_global": {"valor": 20.3353, "umbral": 27.2929, "cumple": True},
        "significativa": {"valor": 0.1664, "umbral": 0.05, "cumple": False},
    }
    return {
        "regla": {
            "incumbente": "sistema_actual",
            "alfa": 0.05,
            "margen_resto_pp": 0.5,
            "sesgo_max_pct": 5.0,
            "criterios": {"gana_global": "WAPE menor", "significativa": "DM con p < α"},
        },
        "campeon": "sistema_actual",
        "mejor_baseline": None,
        "cortes": {"total": 2, "resto": 1},
        "modelos": {
            "sistema_actual": {
                "wape_semanal_global_pct": 27.2929,
                "wape_semanal_resto_pct": 10.9786,
                "sesgo_semanal_max_abs_pct": 2.9144,
                "sesgo_semanal_peor_subgrupo": "categoria=Hogar",
                "criterios": None,
                "pasa": None,
            },
            "lightgbm": {
                "wape_semanal_global_pct": 20.3353,
                "wape_semanal_resto_pct": 11.3004,
                "sesgo_semanal_max_abs_pct": 4.6836,
                "sesgo_semanal_peor_subgrupo": "categoria=Abarrotes",
                "criterios": criterios,
                "pasa": False,
            },
        },
        "pruebas_dm": [
            {
                "a": "lightgbm",
                "b": "sistema_actual",
                "segmento": "global",
                "n_cortes": 2,
                "diferencia_wape_pp": -6.9576,
                "media_d": -1.0,
                "estadistico": -1.4,
                "p_valor": 0.166423,
                "a_favor_de": "lightgbm",
            }
        ],
        "justificacion": (
            "Ningún retador cumple las cuatro condiciones, así que sistema_actual sigue como campeón. El de "
            "menor WAPE semanal global, lightgbm (20.34 % contra 27.29 %), falla porque la diferencia no es "
            "significativa (p = 0.166, α = 0.05)."
        ),
        "sensibilidad": [{"margen_resto_pp": 0.5, "alfa": 0.05, "pasan": [], "campeon": "sistema_actual"}],
    }


def _semana_e7(origen: str, periodo: str, alertas: list, recomendaciones: list) -> dict:
    return {
        "origen": origen,
        "periodo": periodo,
        "desde": origen,
        "hasta": origen,
        "evento": "resto",
        "dias_especiales": [],
        "desempeno": None,
        "alertas": alertas,
        "recomendaciones": recomendaciones,
    }


def _monitor_e7() -> dict:
    alerta = {
        "capa": "calidad",
        "chequeo": "tienda_sin_datos",
        "segmento": "tienda",
        "nivel": "STR_002",
        "modelo": None,
        "medida": "pct_sin_units_sold",
        "valor": 45.2381,
        "referencia": 3.0235,
        "umbral": 0.000125,
        "p_valor": 1e-18,
        "severidad": "alerta",
        "clasificacion": "inesperada",
        "cuando": "al_cerrar",
    }
    captura = {
        "codigo": "revisar_captura",
        "texto": "revisar la captura de las tiendas",
        "por": ["tienda=STR_002"],
    }
    # Más de MAX_POR alertas: la herramienta las cuenta por chequeo, sin identificadores de tienda
    reentrenar = {
        "codigo": "reentrenar",
        "texto": "reentrenar el segmento",
        "por": [
            "region=Centro · nivel_demanda",
            "region=Norte · nivel_demanda",
            "tienda=STR_003",
            "tienda=STR_004",
            "tienda=STR_005",
        ],
    }
    capas = {
        "calidad": {"alertas": 1, "inesperadas": 1, "esperadas": 0, "vigilar": 0},
        "entradas": {"alertas": 12, "inesperadas": 7, "esperadas": 5, "vigilar": 2},
        "desempeno": {"alertas": 0, "inesperadas": 0, "esperadas": 0, "vigilar": 0},
    }
    return {
        "campeon": "sistema_actual",
        "retador": "lightgbm",
        "configuracion": {
            "referencia": "los 56 días que terminan en T",
            "segmentos": "global y cada nivel",
            "esperadas_por_calendario": "nivel, ticket y desempeño",
            "banda_wape": {"sistema_actual": {"semanas": 8, "mediana_pct": 12.1619, "banda_pct": 18.2429}},
            "umbrales": {
                "psi_alerta": 0.25,
                "psi_vigilar": 0.1,
                "banda_wape": 1.5,
                "sesgo_localizado_pp": 10.0,
            },
        },
        "semanas": [
            _semana_e7("2023-10-26", "calibracion", [], [{"codigo": "sin_accion", "texto": "—", "por": []}]),
            _semana_e7(ORIGENES_E7[0], "backtest", [], [{"codigo": "sin_accion", "texto": "—", "por": []}]),
            _semana_e7(ORIGENES_E7[1], "backtest", [alerta], [captura, reentrenar]),
        ],
        "resumen": {
            "calibracion": {
                "semanas": 1,
                "semanas_con_dia_especial": 0,
                "por_capa": capas,
                "recomendaciones": {"sin_accion": 1},
            },
            "backtest": {
                "semanas": 2,
                "semanas_con_dia_especial": 0,
                "por_capa": capas,
                "recomendaciones": {"sin_accion": 1, "revisar_captura": 1},
            },
            "falsas_alarmas_calibracion": {"tienda_sin_datos": 0},
        },
        "vigilancia_retador": {
            "por_corte": [
                {
                    "corte": ORIGENES_E7[1],
                    "cortes_cerrados": 2,
                    "wape_acumulado_pct": {"sistema_actual": 27.2929, "lightgbm": 20.3353},
                    "cortes_gana_retador": 2,
                    "cortes_necesarios": 34,
                    "umbral_p": 0.002941,
                    "reevaluar": False,
                }
            ],
            "final": {"p_valor": 0.166423, "a_favor_de": "lightgbm", "cortes_necesarios": 34},
            "minima": {"corte": ORIGENES_E7[1], "p_valor": 0.138471},
            "coincide_con_seleccion": True,
        },
        "validacion": {
            "todos_detectados": True,
            "escenarios": [
                {
                    "escenario": "falla_pos",
                    "descripcion": "units_sold en nulo",
                    "origen": "2023-10-26",
                    "periodo": "calibracion",
                    "detectado": True,
                    "recomendacion_esperada": "revisar_captura",
                    "recomendaciones": ["revisar_captura"],
                    "alertas_inesperadas": 1,
                }
            ],
        },
    }


def _impacto_e7() -> dict:
    politicas = ("puntual_sistema_actual", "newsvendor_sistema_actual", "newsvendor_lightgbm")
    costos = {
        "global": (352148369.79, 212717514.08, 193003528.19),
        "buen_fin": (150.2e6, 114.31e6, 101.9e6),
        "resto": (95094418.24, 32324606.58, 32413756.78),
    }
    impacto = []
    for nivel, por_politica in costos.items():
        segmento = "global" if nivel == "global" else "evento"
        for politica, costo in zip(politicas, por_politica, strict=True):
            impacto.append(
                {
                    "segmento": segmento,
                    "nivel": nivel,
                    "politica": politica,
                    "pares": 90,
                    "costo": costo,
                    "sobreinventario_pct": 14.4861,
                    "quiebre_pct": 12.8067,
                    "fill_rate_pct": 87.1933,
                    "nivel_servicio_pct": 49.6805,
                    "fc_objetivo_pct": 91.9673,
                    "stock_sobre_demanda": 1.0168,
                }
            )
    prueba = {"n_cortes": 2, "media_d": -1.0, "estadistico": -2.6}
    return {
        "campeon": "sistema_actual",
        "retador": "lightgbm",
        "politicas": {p: "S = Σŷ × q" for p in politicas},
        "supuestos": {
            "costos": {
                "Bebidas": {
                    "margen_bruto": 0.25,
                    "mantener_por_semana": 0.015,
                    "fraccion_critica": 0.9434,
                    "nota": "",
                }
            },
            "fracciones_sensibilidad": [0.8, 0.9],
        },
        "impacto": impacto,
        "pruebas_costo": [
            {
                **prueba,
                "a": politicas[2],
                "b": politicas[1],
                "segmento": "global",
                "costo_a": 193003528.19,
                "costo_b": 212717514.08,
                "diferencia_pct": -9.2677,
                "p_valor": 0.196263,
                "a_favor_de": politicas[2],
            },
            {
                **prueba,
                "a": politicas[1],
                "b": politicas[0],
                "segmento": "global",
                "costo_a": 212717514.08,
                "costo_b": 352148369.79,
                "diferencia_pct": -39.5943,
                "p_valor": 0.018591,
                "a_favor_de": politicas[1],
            },
        ],
        "sensibilidad": {
            "orden_base": list(reversed(politicas)),
            "orden_se_mantiene": True,
            "fracciones": [
                {
                    "fraccion_critica": 0.8,
                    "orden": list(reversed(politicas)),
                    "global": {p: {"costo": c} for p, c in zip(politicas, costos["global"], strict=True)},
                }
            ],
        },
        "conclusion": (
            "Con los costos supuestos, newsvendor_sistema_actual cuesta 39.6 % menos que "
            "puntual_sistema_actual (Diebold-Mariano p = 0.019)."
        ),
        "definiciones": {
            "costo": "Σ Cu · quiebre + Co · sobreinventario",
            "fill_rate_pct": "100 · (1 − quiebre / demanda)",
            "nivel_servicio_pct": "% de pares sin quiebre",
            "sobreinventario": "Σ max(S − y, 0)",
            "quiebre": "Σ max(y − S, 0)",
            "q": "cuantil ponderado",
        },
    }


def _estimacion_e7(efecto: float, ic: tuple[float, float], p: float, **extra) -> dict:
    return {
        "beta": efecto / 100,
        "ee": 0.001,
        "efecto_pct": efecto,
        "ic95_pct": list(ic),
        "p_valor": p,
        **extra,
    }


def _promociones_e7() -> dict:
    por_descuento = [
        {
            "descuento": d,
            "origen": "rejilla",
            "lift_equilibrio_pct": 100 * d / (0.25 - d),
            "veredicto": "no_se_paga",
        }
        for d in (0.05, 0.1)
    ]
    return {
        "muestra": {
            "filas": 1100,
            "series": 6,
            "tiendas": 3,
            "desde": "2023-01-01",
            "hasta": "2023-07-15",
            "dias_promocion": 220,
            "tasa_promocion_pct": 20.0,
        },
        "balance": {
            "b1": {"descripcion": "MLP de has_promotion", "p_valor": 0.821471, "cumple": True},
            "b3": {"descripcion": "promoción futura", "p_valor": 0.346612, "cumple": True},
            "familia": {
                "n_pruebas": 3,
                "alfa": 0.05,
                "p_holm_min": 0.0,
                "rechazadas": ["B5:conteo_inconsistente"],
                "pruebas": [
                    {
                        "prueba": "B5:conteo_inconsistente",
                        "p_holm": 0.0,
                        "tasa_con_promocion_pct": 100.0,
                        "tasa_sin_promocion_pct": 0.0,
                    }
                ],
            },
            "se_sostiene": False,
            "fallan": ["B5:conteo_inconsistente"],
            "asignacion_se_sostiene": True,
            "pruebas_de_medicion": ["B5:conteo_inconsistente"],
        },
        "efecto_principal": {
            "modelo": "log(units_sold) = β · has_promotion + α + γ",
            **_estimacion_e7(0.0472239, (-0.249096, 0.344424), 0.752217),
            "p_aleatorizacion": 0.771614,
            "inferencia_principal": "agrupada_por_tienda",
            "p_valor_principal": 0.752217,
            "significativo": False,
        },
        "canales": {
            "unidades": {
                "resultado": "log(units_sold)",
                "secundario": False,
                **_estimacion_e7(0.0472239, (-0.249096, 0.344424), 0.752217),
            },
            "transacciones": {
                "resultado": "log(total_transactions)",
                "secundario": True,
                **_estimacion_e7(21.8852, (21.6902, 22.0804), 9.7e-116),
                "advertencia": "el registro difiere con promoción",
            },
            "transacciones_efectivo_tarjeta": {
                "resultado": "log(cash + card)",
                "secundario": True,
                **_estimacion_e7(-0.121628, (-0.280313, 0.0373092), 0.131628),
            },
        },
        "robustez": {
            "sin_efectos_fijos": {
                "descripcion": "diferencia de medias",
                **_estimacion_e7(0.319543, (-0.584351, 1.23166), 0.484988),
            },
            "aleatorizacion": {"descripcion": "permutaciones", "p_valor": 0.771614},
        },
        "calibracion_aa": {
            "tasa_pct": 5.2,
            "intervalo_99_pct": [2.6, 7.6],
            "dentro": True,
            "inferencia_principal": "agrupada_por_tienda",
        },
        "heterogeneidad": {
            "categoria": {
                "grupos": {
                    "Abarrotes": _estimacion_e7(
                        -0.312731, (-1.17107, 0.553062), 0.473066, p_holm=1.0, significativo_holm=False
                    ),
                    "Bebidas": _estimacion_e7(
                        1.45283, (0.60629, 2.30649), 0.000973656, p_holm=0.0087629, significativo_holm=True
                    ),
                },
                "igualdad": {"p_valor": 0.0285362, "hay_heterogeneidad": True},
            },
            "familia_holm": {"pruebas": 2, "alfa": 0.05, "significativos": ["categoria:Bebidas"]},
        },
        "dinamica": {
            "suma_rezagos": _estimacion_e7(-0.553747, (-1.49381, 0.395288), 0.248051),
            "neto": _estimacion_e7(-0.53953, (-1.5501, 0.481412), 0.294909),
            "adelanto": False,
        },
        "cruzado": {
            "otras_categorias": _estimacion_e7(0.0597017, (-0.0807746, 0.200375), 0.400336),
            "interpretacion": "sin_efecto_detectable",
        },
        "potencia": {
            "diseno_actual": {
                "alfa": 0.05,
                "potencia": 0.8,
                "global": {"mde_pct": 0.423616},
                "categoria": {"Bebidas": {"mde_pct": 1.20129}},
            },
            "diseno_semanal": {
                "descripcion": "promoción por semana",
                "semanas": 26,
                "mde_actual": {"mde_pct": 0.416623},
                "por_semanas": [{"semanas": 4, "mde_pct": 1.93329}],
                "semanas_necesarias": [{"efecto_pct": 2.0, "semanas": 4}],
            },
        },
        "negocio": {
            "descuento_implicito": {
                "aplica": False,
                "descuento": None,
                "regla": "IC95 del precio abajo de cero",
            },
            "por_categoria": {
                "Bebidas": {
                    "margen_bruto": 0.25,
                    "precio_lista": 74.8331,
                    "lift_pct": 1.45283,
                    "lift_ic95_pct": [0.60629, 2.30649],
                    "incrementales_por_dia_promocion": 25.5949,
                    "valor_maximo_sin_descuento_por_dia": 478.836,
                    "por_descuento": por_descuento,
                }
            },
            "por_descuento": [
                {
                    "descuento": d,
                    "origen": "rejilla",
                    "se_paga": [],
                    "no_se_paga": ["Abarrotes", "Bebidas"],
                    "indeterminado": [],
                }
                for d in (0.05, 0.1)
            ],
        },
        "supuestos": {
            "negocio": {"margen_bruto": {"Bebidas": 0.25}, "descuentos": [0.05, 0.1]},
            "estimacion": "log-OLS con efectos fijos",
            "no_anticipacion": "B3",
            "sin_contaminacion": "sin contagio",
        },
        "conclusion": {
            "aleatorizacion_se_sostiene": False,
            "efecto_principal_pct": 0.0472239,
            "texto": "La promoción cambia las unidades en +0.05 % (IC95 de -0.25 a +0.34 %; p = 0.752).",
        },
    }


def _calidad_e7() -> dict:
    return {
        "contratos": {"transacciones": "ok", "tiendas": "ok", "calendario": "ok", "uniones": "ok"},
        "dimensiones": {
            "filas": 1176,
            "tiendas": 3,
            "categorias": 2,
            "dias": 196,
            "series": 6,
            "desde": "2023-01-01",
            "hasta": "2023-07-15",
        },
        "malla": {
            "esperadas": 1180,
            "observadas": 1176,
            "faltantes": 4,
            "bloques": [
                {"store_id": "STR_002", "desde": "2023-02-01", "hasta": "2023-02-02", "dias": 2, "filas": 4}
            ],
        },
        "conteos": {
            "regla": "cash + card == total",
            "filas_evaluables": 1100,
            "inconsistentes": 220,
            "pct_inconsistentes": 20.0,
            "uniformidad_pct": {"tienda": {"min": 18.5, "max": 21.5}},
            "avg_ticket": {"sobreestimacion_media_pct_en_inconsistentes": 21.7},
        },
        "nulos": {
            "units_sold": {"nulos": 35, "pct": 2.98, "patron": "disperso"},
            "replenishment_signal": {
                "nulos": 12,
                "pct": 1.02,
                "patron": "bloque",
                "fechas": ["2023-01-01", "2023-01-02"],
            },
        },
        "fuga_replenishment_signal": {
            "criterio": "correlación intraserie",
            "mejor_ventana": {"dias": 7, "desde": -5, "hasta": 1, "corr": 0.9312},
            "factor": 0.5212,
            "pesos_por_desfase": {"-1": 0.061, "+0": 0.058, "+1": 0.055},
            "corr_con_units_sold": {"senal_del_mismo_dia": 0.5375},
            "incluye_el_dia": True,
            "usa_informacion_futura": True,
            "rezago_minimo_desde_el_origen": 1,
        },
        "decisiones": ["replenishment_signal: con origen T solo se usan valores fechados en T - 1 o antes."],
    }


def _eda_e7() -> dict:
    return {
        "demanda": {
            "nivel_medio_por_categoria": {"Abarrotes": 2271.5, "Bebidas": 1765.0},
            "nivel_medio_por_formato": {"Supercenter": 1919.6},
            "participacion_pct_por_categoria": {"Abarrotes": 56.27, "Bebidas": 43.73},
            "tendencia": {
                "crecimiento_interanual_pct": 9.55,
                "mes_mas_alto": {"mes": "2023-07", "indice": 1.2},
            },
            "estacionalidad_semanal": {"perfil_dia_semana": {"lunes": 0.75, "sábado": 1.395}},
            "dispersion_series": {"razon_serie_max_min": 6.2, "cv_entre_series": 0.4},
            "efectos_calendario": {
                "buen_fin": None,
                "quincena": 1.3845,
                "fin_de_semana": 1.5283,
                "navidad": None,
                "semana_santa": 0.6641,
            },
            "dispersion_efectos_por_categoria": {
                "quincena": {"min": 1.3618, "max": 1.3944, "rango_pct": 2.39}
            },
            "dias_por_efecto": {"quincena": 12, "fin_de_semana": 56},
            "dias_atipicos": {"umbral": 0.3, "fechas": []},
            "festivos": [
                {"fecha": "2023-01-01", "nombre": "Año Nuevo", "multiplicador": 0.427, "en_navidad": False}
            ],
        },
        "promociones": {
            "tasa_global_pct": 19.98,
            "rango_tasa_pp": {"formato": 0.12},
            "autocorrelacion": {"lag_1": -0.0026},
            "lift_ingenuo_pct": {"Bebidas": 1.84},
            "lift_ingenuo_ic95_pct": {"Bebidas": [0.05, 3.62]},
            "categorias_con_ic95_sin_cero": ["Bebidas"],
            "nota": "descriptivo",
        },
        "tiendas": {
            "tiendas": 3,
            "por_formato": {"Supercenter": 1, "Bodega": 1, "Express": 1},
            "por_region": {"Centro": 3},
            "por_nse": {"C": 3},
            "corr_nivel_m2": 0.828,
        },
        "sistema_actual": {
            "protocolo": {"desde": "2023-03-01"},
            "wape_pct": {
                "senal_T_factor_ajustado": 18.69,
                "senal_T_menos_1_factor_ajustado": 20.11,
                "ma7": 21.59,
            },
            "sesgo_pct": {"ma7": 0.23},
            "wape_por_evento_pct": {"ma7": {"resto": 16.52}},
            "aumento_wape_al_quitar_fuga_pp": 1.52,
            "ventaja_sobre_ma7_sin_fuga_pp": 1.48,
        },
        "implicaciones": ["La señal de reposición solo se usa rezagada (T−1)."],
    }


def _pronosticos_e7() -> pl.DataFrame:
    filas = []
    for modelo in MODELOS_E7:
        for origen in ORIGENES_E7:
            inicio = date.fromisoformat(origen)
            for tienda in ("STR_001", "STR_002"):
                for categoria in ("Abarrotes", "Bebidas"):
                    for h in range(1, 8):
                        yhat = 100.0 + h + (5 if modelo == "lightgbm" else 0)
                        filas.append(
                            {
                                "modelo": modelo,
                                "origen": inicio,
                                "store_id": tienda,
                                "category": categoria,
                                "date": inicio + timedelta(days=h),
                                "h": h,
                                "y": 101.0 + h,
                                "yhat": yhat,
                                "p10": None if modelo == "sistema_actual" else 0.8 * yhat,
                                "p90": None if modelo == "sistema_actual" else 1.2 * yhat,
                            }
                        )
    return pl.DataFrame(filas)


def escribir_artefactos_e7(carpeta: Path) -> Path:
    """Los siete artefactos que leen las herramientas de E7, con la forma de los reales y pocas cifras."""
    contenido = {
        "calidad/reporte.json": _calidad_e7(),
        "eda/resumen.json": _eda_e7(),
        "backtest/metricas.json": _metricas_e7(),
        "modelos/seleccion.json": _seleccion_e7(),
        "drift/monitor.json": _monitor_e7(),
        "decision/impacto.json": _impacto_e7(),
        "causal/promociones.json": _promociones_e7(),
    }
    for relativa, datos in contenido.items():
        ruta = carpeta / relativa
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False))
    _pronosticos_e7().write_parquet(carpeta / "backtest" / "pronosticos.parquet")
    return carpeta


@pytest.fixture
def artefactos_e7(tmp_path: Path) -> Path:
    return escribir_artefactos_e7(tmp_path / "artifacts")


@pytest.fixture(autouse=True)
def sin_red_ni_keys(monkeypatch):
    """Ninguna prueba sale a la red ni lee una key: las variables vacías ganan al `.env` real."""
    from pydantic_ai import models

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    for variable in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(variable, "")
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    monkeypatch.setenv("DEMANDA_ENV_FILE", str(SIN_ENV))


INVENTADA = "123456.789"  # una cifra que no está en ningún artefacto sintético


def _primer_numero(x) -> str | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, int | float):
        return str(x)
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, list):
        for v in x:
            if (n := _primer_numero(v)) is not None:
                return n
    return None


def cita_real(messages, info) -> str:
    """La primera cifra de las respuestas de las herramientas; sin herramientas (el orquestador), la primera
    cifra de la evidencia que recibió en el mensaje."""
    from pydantic_ai.messages import ToolReturnPart, UserPromptPart

    for m in messages:
        for p in m.parts:
            if isinstance(p, ToolReturnPart) and (n := _primer_numero(p.content)) is not None:
                return n
    for m in messages:
        for p in m.parts:
            texto = p.content if isinstance(p, UserPromptPart) and isinstance(p.content, str) else ""
            if encontrada := re.search(r'"cifra": "([^"]+)"', texto):
                return encontrada.group(1)
    raise AssertionError("no hay cifras en el historial")


def inventa(messages, info) -> str:
    return INVENTADA


def inventa_y_corrige(messages, info) -> str:
    """Inventa en el primer intento y cita una cifra real cuando recibe el reintento."""
    from pydantic_ai.messages import UserPromptPart

    reintento = any(
        isinstance(p, UserPromptPart) and "El validador rechazó" in str(p.content)
        for m in messages
        for p in m.parts
    )
    return cita_real(messages, info) if reintento else INVENTADA


class ModeloSimulado:
    """Un `FunctionModel` de prueba. Al empezar cada intento llama a todas las herramientas del agente;
    después entrega `insights` insights que citan la cifra que da `citar`. Reporta `tokens` (entrada,
    salida) por respuesta; con `rechazo`, toda respuesta es un rechazo por seguridad. Guarda en `vistos`
    los mensajes que recibió en cada solicitud."""

    def __init__(
        self,
        citar=cita_real,
        tokens: tuple[int, int] = (1_000, 500),
        rechazo: bool = False,
        insights: int = 2,
    ) -> None:
        self.citar, self.tokens, self.rechazo, self.insights = citar, tokens, rechazo, insights
        self.solicitudes = 0
        self.vistos: list[list] = []

    def __call__(self, messages, info):
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart
        from pydantic_ai.usage import RequestUsage

        self.solicitudes += 1
        self.vistos.append(list(messages))
        uso = RequestUsage(input_tokens=self.tokens[0], output_tokens=self.tokens[1])
        if self.rechazo:
            detalles = {"finish_reason": "refusal", "refusal_category": "cyber"}
            return ModelResponse(
                parts=[], usage=uso, finish_reason="content_filter", provider_details=detalles
            )
        inicio = any(isinstance(p, UserPromptPart) for p in messages[-1].parts)
        if info.function_tools and inicio:
            llamadas = [
                ToolCallPart(t.name, {}, tool_call_id=f"c{i}") for i, t in enumerate(info.function_tools)
            ]
            return ModelResponse(parts=llamadas, usage=uso)
        cifra = self.citar(messages, info)
        herramienta = info.function_tools[0].name if info.function_tools else "get_data_quality"
        insight = {
            "titulo": "Un hallazgo",
            "hallazgo": f"La cifra que importa es {cifra}.",
            "evidencia": [{"cifra": cifra, "que_mide": "la cifra", "herramienta": herramienta}],
            "accion": "Actuar en consecuencia.",
            "confianza": "media",
        }
        salida = {"insights": [{**insight, "titulo": f"Hallazgo {n}"} for n in range(1, self.insights + 1)]}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, salida)], usage=uso)
        return ModelResponse(parts=[TextPart(json.dumps(salida))], usage=uso)

    def modelo(self):
        from pydantic_ai.models.function import FunctionModel

        return FunctionModel(self)


@pytest.fixture
def config_e7(artefactos_e7: Path):
    """Config sin `.env`, sobre los artefactos sintéticos, con el modelo por omisión."""
    from demanda.config import Config

    return Config(_env_file=None, artifacts_dir=artefactos_e7, llm_model="anthropic:claude-opus-5-5")


@pytest.fixture
def simulado() -> SimpleNamespace:
    """`ModeloSimulado` y las formas de citar, para las pruebas de los agentes."""
    return SimpleNamespace(
        Modelo=ModeloSimulado,
        cita_real=cita_real,
        inventa=inventa,
        inventa_y_corrige=inventa_y_corrige,
        inventada=INVENTADA,
    )
